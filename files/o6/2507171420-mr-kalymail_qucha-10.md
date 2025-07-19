# MR 10 – Fix(spamrock.c#send_shiva_notif): Split notif by domains

## Description
Il s'agit de grouper par domaines les destinataires pour émettre les notifications aux tenants dédié à ces domaines

## Fichiers modifiés
- M src/constants.h
- M src/spamrock.c
- M src/spamrock.h
- M src/types.h
- M src/utils.c
- M src/utils.h
- M tests/Makefile
- M tests/test_spamrock.c
- A tests/test_utils.c
- M .gitignore
- M ChangeLog

## Diff complet

### src/constants.h
```diff
@@ -4,7 +4,7 @@
 #define BUFFER_SIZE 6000
 #define MAX_FILES 1000
 
-#define VERSION "0.12.0"
+#define VERSION "0.13.0"
 
 #define PORT_IN 10025
 #define PORT_OUT 1025

```

### src/spamrock.c
```diff
@@ -17,11 +17,9 @@
 #include <stdbool.h>
 #include <sys/stat.h>
 #include <sys/time.h>
-#include <curl/curl.h>
 #include <arpa/inet.h>
 #include <sys/types.h>
 #include <sys/socket.h>
-#include <sys/inotify.h>
 
 #include "types.h"
 #include "utils.h"
@@ -38,16 +36,13 @@ static void print_logo();
 static void listen_smtp();
 void create_logs_directory();
 static void listen_output_dir();
-static int send_shiva_notif(Gauth dto);
-static int handle_http_error(CURLcode res);
+int send_shiva_notif(Gauth dto);
 static void send_email(const char *filename);
-static int handle_http_status(long http_code);
 static int handle_smtp_connection(int client_socket);
 char* remove_substrings(char buffer[], const char *substrings[]);
 char* extract_text(const char* str, char start_char, char end_char);
 int receive_and_parse_email(int client_socket, FILE *file, Gauth *gauth);
 static int send_smtp_command(int sockfd, const char *cmd, const char *expected_code);
-static size_t writeMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);
 static void find_eml_files(const char *directory, char files[MAX_FILES][256], int *file_count);
 
 /* ----------------------------------------------------------------------------------
@@ -817,41 +812,21 @@ static void print_logo() {
   }
 }
 
-static int handle_http_error(CURLcode res) {
-  switch (res) {
-    case CURLE_OK:
-      return 0;
-      break;
-    case CURLE_OPERATION_TIMEDOUT:
-      logger(ERROR, "handle_http_error", "Error: Timeout occurred");
-      return 1;
-      break;
-    default:
-      logger(ERROR, "handle_http_error", "Error: %s", curl_easy_strerror(res));
-      return 1;
-  }
-}
-
-static int handle_http_status(long http_code) {
-  switch (http_code) {
-    case 200:
-      return 0;
-      break;
-    case 401:
-      logger(ERROR, "handle_http_status", "Error 401: Unauthorized");
-      return 1;
-      break;
-    case 404:
-      logger(ERROR, "handle_http_status", "Error 404: Not Found");
-      return 1;
-      break;
-    default:
-      logger(ERROR, "handle_http_status", "HTTP Error: %ld", http_code);
-      return 1;
-  }
-}
-
-static int send_shiva_notif(Gauth dto) {
+/**
+ * Sends a notification to the Shiva API with email metadata.
+ *
+ * This function constructs a JSON payload containing information extracted from
+ * the received email (message_id, from, to, subject, file name, etc.) and sends it
+ * to the configured Shiva API endpoint (global_config.url_shiva).
+ *
+ * If the message_id is missing or equals the default "message_id", a new one is generated
+ * using a timestamp to ensure uniqueness.
+ *
+ * @param dto A Gauth struct containing metadata about the email to report.
+ * @return 0 if the request was successfully sent and acknowledged (HTTP 200),
+ *         or a non-zero error code if the operation failed (e.g., network issue, invalid response).
+ */
+int send_shiva_notif(Gauth dto) {
   logger(DEBUG, "send_shiva_notif", "Start");
   logger(DEBUG, "send_shiva_notif", "dto.message_id => %s", dto.message_id);
   logger(DEBUG, "send_shiva_notif", "dto.origin => %s", dto.origin);
@@ -860,29 +835,17 @@ static int send_shiva_notif(Gauth dto) {
   logger(DEBUG, "send_shiva_notif", "dto.subject => %s", dto.subject);
   logger(DEBUG, "send_shiva_notif", "dto.file_name => %s", dto.file_name);
 
-  // If dto.message_id is still "message_id" or empty, generate a new one
   if (strcmp(dto.message_id, "message_id") == 0 || dto.message_id[0] == '\0') {
     char generated_id[128];
-    snprintf(generated_id, sizeof(generated_id), "spamrock-%ld@spamrockd.org", (long)time(NULL));  // Using the current timestamp for uniqueness
+    snprintf(generated_id, sizeof(generated_id), "spamrock-%ld@spamrockd.org", (long)time(NULL));
 
     strncpy(dto.message_id, generated_id, sizeof(dto.message_id));
-    // Ensure null termination
     dto.message_id[sizeof(dto.message_id) - 1] = '\0';
 
     logger(DEBUG, "send_shiva_notif", "dto.message_id was default; generated => %s", dto.message_id);
   }
 
-  CURL *curl;
-  CURLcode res;
-  long http_code = 0;
-  struct MemoryStruct chunk;
-  int response = 0;
-
-  chunk.memory = malloc(1);  // initial allocation
-  chunk.size = 0;            // no data at this point
-
-  // Constructing the JSON data string using sprintf
-  char json_data[BUFFER_SIZE * 10]; // Allocate enough space for the JSON data
+  char json_data[BUFFER_SIZE * 10];
 
   snprintf(json_data, sizeof(json_data),
             "{"
@@ -905,46 +868,27 @@ static int send_shiva_notif(Gauth dto) {
   logger(DEBUG, "send_shiva_notif", "json_data => %s", json_data);
   logger(DEBUG, "send_shiva_notif", "global_config.url_shiva => %s", global_config.url_shiva);
 
-  // Building the URL by appending send_to=dto.to
-  char url_with_to[13000];
-  snprintf(url_with_to, sizeof(url_with_to), "%s?send_to=%s", global_config.url_shiva, dto.to);
+  DomainGroup groups[20];
+  int nb_domains = group_recipients_by_domain(dto.to, groups, 20);
 
-  logger(DEBUG, "send_shiva_notif", "url_with_to => %s", url_with_to);
+  int global_status = 0;
 
-  curl_global_init(CURL_GLOBAL_DEFAULT);
-  curl = curl_easy_init();
-  if (curl) {
-    struct curl_slist *headers = NULL;
-    headers = curl_slist_append(headers, "Content-Type: application/json");
-    headers = curl_slist_append(headers, "Accept: application/json");
+  for (int i = 0; i < nb_domains; i++) {
+    logger(DEBUG, "send_shiva_notif", "Sending to domain: %s", groups[i].domain);
+    logger(DEBUG, "send_shiva_notif", "Recipients: %s", groups[i].recipients);
 
-    curl_easy_setopt(curl, CURLOPT_URL, url_with_to);
-    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
-    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data);
-    curl_easy_setopt(curl, CURLOPT_TIMEOUT, TIMEOUT);
-    // Configure the callback to capture the data received into chunk.memory
-    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeMemoryCallback);
-    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
+    char url_with_to[13000];
+    snprintf(url_with_to, sizeof(url_with_to), "%s?send_to=%s", global_config.url_shiva, groups[i].recipients);
 
-    res = curl_easy_perform(curl);
+    int result = post_json_to_shiva(url_with_to, json_data);
 
-    if (res != CURLE_OK) {
-      response = handle_http_error(res);
-    } else {
-      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
-      response = handle_http_status(http_code);
-      logger(DEBUG, "send_shiva_notif", "Received data: %s", chunk.memory);
+    if (result != 0) {
+      logger(ERROR, "send_shiva_notif", "Failed to send to %s", groups[i].domain);
+      global_status = result;
     }
-
-    curl_slist_free_all(headers);
-    curl_easy_cleanup(curl);
-    free(chunk.memory);
-  } else {
-    response = 1;
   }
 
-  curl_global_cleanup();
-  return response;
+  return global_status;
 }
 
 /**
@@ -985,33 +929,6 @@ char* extract_text(const char* str, char start_char, char end_char) {
   return extracted_text;
 }
 
-/**
- * Callback function to capture received HTTP response data.
- *
- * @param contents Pointer to the received data.
- * @param size Size of one data chunk.
- * @param nmemb Number of data chunks.
- * @param userp Pointer to the MemoryStruct where data should be stored.
- * @return The number of bytes processed.
- */
-static size_t writeMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
-  size_t realsize = size * nmemb;
-  struct MemoryStruct *mem = (struct MemoryStruct *)userp;
-
-  char *ptr = realloc(mem->memory, mem->size + realsize + 1);
-  if(ptr == NULL) {
-    logger(ERROR, "writeMemoryCallback", "Not enough memory (realloc returned NULL)");
-    return 0;
-  }
-
-  mem->memory = ptr;
-  memcpy(&(mem->memory[mem->size]), contents, realsize);
-  mem->size += realsize;
-  mem->memory[mem->size] = 0;
-
-  return realsize;
-}
-
 /**
  * Removes all occurrences of specified substrings from the input buffer.
  *

```

### src/spamrock.h
```diff
@@ -8,6 +8,7 @@
 #define SPAMROCK_H
 
 char* get_current_time();
+int send_shiva_notif(Gauth dto);
 char* decode_mime(const char* input);
 char* extract_message_id(const char *line);
 int has_eml_extension(const char *filename);

```

### src/types.h
```diff
@@ -33,4 +33,9 @@ struct MemoryStruct {
 };
 
 extern char* global_mode;
-extern Config global_config;
\ No newline at end of file
+extern Config global_config;
+
+typedef struct {
+  char domain[256];
+  char recipients[BUFFER_SIZE];
+} DomainGroup;
\ No newline at end of file

```

### src/utils.c
```diff
@@ -6,7 +6,9 @@
 #include <ctype.h>
 #include <stdlib.h>
 #include <string.h>
+#include <search.h>
 #include <sys/time.h>
+#include <curl/curl.h>
 
 #include "types.h"
 #include "config.h"
@@ -18,12 +20,15 @@
 char* get_current_time();
 char* get_current_date();
 char *trim_whitespace(char *str);
+int handle_http_error(CURLcode res);
 char* decode_mime(const char* input);
+int handle_http_status(long http_code);
 int has_eml_extension(const char *filename);
 char* extract_message_id(const char *email);
 char* parse_subject(char* lines[], int line_count);
 char* decode_mime_single_block(const char* encoded);
 void get_email_filename(char *filename, size_t max_len);
+size_t writeMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);
 
 /* -----------------------------------------------------------------------
  *  IMPLEMENTATION
@@ -520,4 +525,167 @@ char* to_uppercase(const char *input) {
   result[len] = '\0';
 
   return result;
-}
\ No newline at end of file
+}
+
+/**
+ * Groups email recipients by their domain.
+ *
+ * @param to_field     The input string containing recipients separated by ';'
+ * @param groups       The output array of DomainGroup to fill
+ * @param max_groups   The maximum number of groups to fill
+ * @return The number of unique domains found
+ */
+int group_recipients_by_domain(const char *to_field, DomainGroup *groups, int max_groups) {
+  if (!to_field || !groups || max_groups <= 0) return 0;
+
+  char to_copy[BUFFER_SIZE];
+  strncpy(to_copy, to_field, sizeof(to_copy) - 1);
+  to_copy[sizeof(to_copy) - 1] = '\0';
+
+  int count = 0;
+  char *saveptr = NULL;
+  char *token = strtok_r(to_copy, ";", &saveptr);
+
+  while (token != NULL) {
+    char *clean_token = trim_whitespace(token);
+
+    char *at = strchr(clean_token, '@');
+    if (!at || strlen(at + 1) == 0) {
+      token = strtok_r(NULL, ";", &saveptr);
+      continue;
+    }
+
+    char domain[256];
+    strncpy(domain, at + 1, sizeof(domain) - 1);
+    domain[sizeof(domain) - 1] = '\0';
+
+    int found = 0;
+    for (int i = 0; i < count; i++) {
+      if (strcmp(groups[i].domain, domain) == 0) {
+        strncat(groups[i].recipients, ";", sizeof(groups[i].recipients) - strlen(groups[i].recipients) - 1);
+        strncat(groups[i].recipients, clean_token, sizeof(groups[i].recipients) - strlen(groups[i].recipients) - 1);
+        found = 1;
+        break;
+      }
+    }
+
+    if (!found && count < max_groups) {
+      strncpy(groups[count].domain, domain, sizeof(groups[count].domain) - 1);
+      strncpy(groups[count].recipients, clean_token, sizeof(groups[count].recipients) - 1);
+      groups[count].domain[sizeof(groups[count].domain) - 1] = '\0';
+      groups[count].recipients[sizeof(groups[count].recipients) - 1] = '\0';
+      count++;
+    }
+
+    token = strtok_r(NULL, ";", &saveptr);
+  }
+
+  return count;
+}
+
+#ifndef TEST_BUILD
+int post_json_to_shiva(const char *url, const char *json_data) {
+  CURL *curl;
+  CURLcode res;
+  long http_code = 0;
+  int response = 0;
+
+  struct MemoryStruct chunk;
+  chunk.memory = malloc(1);
+  chunk.size = 0;
+
+  curl_global_init(CURL_GLOBAL_DEFAULT);
+  curl = curl_easy_init();
+  if (curl) {
+    struct curl_slist *headers = NULL;
+    headers = curl_slist_append(headers, "Content-Type: application/json");
+    headers = curl_slist_append(headers, "Accept: application/json");
+
+    curl_easy_setopt(curl, CURLOPT_URL, url);
+    curl_easy_setopt(curl, CURLOPT_HTTPHEADER, headers);
+    curl_easy_setopt(curl, CURLOPT_POSTFIELDS, json_data);
+    curl_easy_setopt(curl, CURLOPT_TIMEOUT, TIMEOUT);
+    curl_easy_setopt(curl, CURLOPT_WRITEFUNCTION, writeMemoryCallback);
+    curl_easy_setopt(curl, CURLOPT_WRITEDATA, (void *)&chunk);
+
+    res = curl_easy_perform(curl);
+    if (res != CURLE_OK) {
+      response = handle_http_error(res);
+    } else {
+      curl_easy_getinfo(curl, CURLINFO_RESPONSE_CODE, &http_code);
+      response = handle_http_status(http_code);
+      logger(DEBUG, "post_json_to_shiva", "Received data: %s", chunk.memory);
+    }
+
+    curl_slist_free_all(headers);
+    curl_easy_cleanup(curl);
+    free(chunk.memory);
+  } else {
+    response = 1;
+  }
+
+  curl_global_cleanup();
+  return response;
+}
+#endif
+
+/**
+ * Callback function to capture received HTTP response data.
+ *
+ * @param contents Pointer to the received data.
+ * @param size Size of one data chunk.
+ * @param nmemb Number of data chunks.
+ * @param userp Pointer to the MemoryStruct where data should be stored.
+ * @return The number of bytes processed.
+ */
+size_t writeMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp) {
+  size_t realsize = size * nmemb;
+  struct MemoryStruct *mem = (struct MemoryStruct *)userp;
+
+  char *ptr = realloc(mem->memory, mem->size + realsize + 1);
+  if(ptr == NULL) {
+    logger(ERROR, "writeMemoryCallback", "Not enough memory (realloc returned NULL)");
+    return 0;
+  }
+
+  mem->memory = ptr;
+  memcpy(&(mem->memory[mem->size]), contents, realsize);
+  mem->size += realsize;
+  mem->memory[mem->size] = 0;
+
+  return realsize;
+}
+
+int handle_http_error(CURLcode res) {
+  switch (res) {
+    case CURLE_OK:
+      return 0;
+      break;
+    case CURLE_OPERATION_TIMEDOUT:
+      logger(ERROR, "handle_http_error", "Error: Timeout occurred");
+      return 1;
+      break;
+    default:
+      logger(ERROR, "handle_http_error", "Error: %s", curl_easy_strerror(res));
+      return 1;
+  }
+}
+
+int handle_http_status(long http_code) {
+  switch (http_code) {
+    case 200:
+      return 0;
+      break;
+    case 401:
+      logger(ERROR, "handle_http_status", "Error 401: Unauthorized");
+      return 1;
+      break;
+    case 404:
+      logger(ERROR, "handle_http_status", "Error 404: Not Found");
+      return 1;
+      break;
+    default:
+      logger(ERROR, "handle_http_status", "HTTP Error: %ld", http_code);
+      return 1;
+  }
+}

```

### src/utils.h
```diff
@@ -1,16 +1,24 @@
 // src\utils.h
 #pragma once
 #include <stddef.h>
+#include <curl/curl.h>
+
+#include "types.h"
 
 char* get_current_time();
 char* get_current_date();
 char *trim_whitespace(char *str);
+int handle_http_error(CURLcode res);
 char* decode_mime(const char* input);
 char* to_uppercase(const char *input);
+int handle_http_status(long http_code);
 int has_eml_extension(const char *filename);
 char* extract_message_id(const char *email);
 char* strip_trailing_newlines(const char *str);
 int contains(const char *str, const char *test);
 char* parse_subject(char* lines[], int line_count);
 char* decode_mime_single_block(const char* encoded);
-void get_email_filename(char *filename, size_t max_len);
\ No newline at end of file
+void get_email_filename(char *filename, size_t max_len);
+int post_json_to_shiva(const char *url, const char *json_data);
+size_t writeMemoryCallback(void *contents, size_t size, size_t nmemb, void *userp);
+int group_recipients_by_domain(const char *to_field, DomainGroup *groups, int max_groups);

```

### tests/Makefile
```diff
@@ -3,7 +3,7 @@
 TARGET=test_spamrock
 
 # Fichiers sources
-SRC=test_spamrock.c ../src/logger.c ../src/utils.c ../src/config.c ../src/spamrock.c  Unity/src/unity.c
+SRC=test_spamrock.c test_utils.c ../src/logger.c ../src/utils.c ../src/config.c ../src/spamrock.c Unity/src/unity.c
 
 # Flags pour compilation avec couverture
 CFLAGS=-Wno-trigraphs -DTEST_BUILD -fprofile-arcs -ftest-coverage -Iunity/

```

### tests/test_spamrock.c
```diff
@@ -673,6 +673,48 @@ void test_to_uppercase_various_cases(void) {
   }
 }
 
+#ifdef TEST_BUILD
+int post_json_to_shiva(const char *url, const char *json_data) {
+    TEST_ASSERT_NOT_NULL(url);
+    TEST_ASSERT_NOT_NULL(json_data);
+
+    TEST_ASSERT_TRUE(strstr(url, "send_to=to@example.com") != NULL);
+
+    TEST_ASSERT_TRUE(strstr(json_data, "\"message_id\":") != NULL);
+    TEST_ASSERT_TRUE(strstr(json_data, "\"from\": \"from@example.com\"") != NULL);
+    TEST_ASSERT_TRUE(strstr(json_data, "\"to\": \"to@example.com\"") != NULL);
+    TEST_ASSERT_TRUE(strstr(json_data, "\"file_name\": \"testfile.eml\"") != NULL);
+    TEST_ASSERT_TRUE(strstr(json_data, "\"subject\": \"Test subject\"") != NULL);
+
+    return 0;
+}
+#endif
+
+void test_send_shiva_notif_simple(void) {
+    strcpy(global_config.url_shiva, "http://fake.local");
+
+    Gauth dto = {
+        .message_id = "message_id",
+        .origin = "origin.local",
+        .from = "from@example.com",
+        .to = "to@example.com",
+        .file_name = "testfile.eml",
+        .subject = "Test subject"
+    };
+
+    int result = send_shiva_notif(dto);
+    TEST_ASSERT_EQUAL_INT(0, result);
+}
+
+/**
+ * test_utils.c
+ */
+void test_group_recipients_by_domain_basic_case(void);
+void test_group_recipients_by_domain_trim_spaces(void);
+void test_group_recipients_by_domain_max_groups_limit(void);
+void test_group_recipients_by_domain_multiple_same_domain(void);
+void test_group_recipients_by_domain_invalid_entry_skipped(void);
+
 /**
  * @brief Main function that runs all tests through Unity.
  */
@@ -736,6 +778,16 @@ int main(void) {
     // Test for to_uppercase
     RUN_TEST(test_to_uppercase_various_cases);
 
+    // send_shiva_notif
+    RUN_TEST(test_send_shiva_notif_simple);
+
+    // From test_utils
+    RUN_TEST(test_group_recipients_by_domain_basic_case);
+    RUN_TEST(test_group_recipients_by_domain_trim_spaces);
+    RUN_TEST(test_group_recipients_by_domain_max_groups_limit);
+    RUN_TEST(test_group_recipients_by_domain_multiple_same_domain);
+    RUN_TEST(test_group_recipients_by_domain_invalid_entry_skipped);
+
     int test_result = UNITY_END();
     clock_t end_time = clock();
 

```

### tests/test_utils.c
```diff
@@ -0,0 +1,100 @@
+// tests/test_utils.c
+#define _GNU_SOURCE
+#include <time.h>
+#include <stdio.h>
+#include <string.h>
+#include <stdlib.h>
+#include <signal.h>
+
+#include "../src/utils.h"
+#include "./Unity/src/unity.h"
+
+/**
+ * Tests that multiple recipients with different domains are correctly grouped.
+ */
+void test_group_recipients_by_domain_basic_case(void) {
+    const char *to_field = "alice@gmail.com;bob@yahoo.com;carol@gmail.com";
+    DomainGroup groups[5] = {0};
+
+    int count = group_recipients_by_domain(to_field, groups, 5);
+    TEST_ASSERT_EQUAL_INT(2, count);
+
+    int found_gmail = 0, found_yahoo = 0;
+    for (int i = 0; i < count; i++) {
+        if (strcmp(groups[i].domain, "gmail.com") == 0) {
+            found_gmail = 1;
+            TEST_ASSERT_EQUAL_STRING("alice@gmail.com;carol@gmail.com", groups[i].recipients);
+        } else if (strcmp(groups[i].domain, "yahoo.com") == 0) {
+            found_yahoo = 1;
+            TEST_ASSERT_EQUAL_STRING("bob@yahoo.com", groups[i].recipients);
+        }
+    }
+
+    TEST_ASSERT_TRUE(found_gmail);
+    TEST_ASSERT_TRUE(found_yahoo);
+}
+
+/**
+ * Tests that invalid entries without proper domains are ignored.
+ */
+void test_group_recipients_by_domain_invalid_entry_skipped(void) {
+    const char *to_field = "bob@;alice@gmail.com;;test@";
+    DomainGroup groups[3] = {0};
+
+    int count = group_recipients_by_domain(to_field, groups, 3);
+    TEST_ASSERT_EQUAL_INT(1, count);
+    TEST_ASSERT_EQUAL_STRING("gmail.com", groups[0].domain);
+    TEST_ASSERT_EQUAL_STRING("alice@gmail.com", groups[0].recipients);
+}
+
+/**
+ * Tests that whitespaces around recipients are trimmed and do not affect grouping.
+ */
+void test_group_recipients_by_domain_trim_spaces(void) {
+    const char *to_field = "   bob@gmail.com  ; alice@yahoo.com ;carol@gmail.com ";
+    DomainGroup groups[3] = {0};
+
+    int count = group_recipients_by_domain(to_field, groups, 3);
+    TEST_ASSERT_EQUAL_INT(2, count);
+
+    for (int i = 0; i < count; i++) {
+        if (strcmp(groups[i].domain, "gmail.com") == 0) {
+            TEST_ASSERT_EQUAL_STRING("bob@gmail.com;carol@gmail.com", groups[i].recipients);
+        } else if (strcmp(groups[i].domain, "yahoo.com") == 0) {
+            TEST_ASSERT_EQUAL_STRING("alice@yahoo.com", groups[i].recipients);
+        }
+    }
+}
+
+/**
+ * Tests that the function respects the max_groups limit and skips extra domains.
+ */
+void test_group_recipients_by_domain_max_groups_limit(void) {
+    const char *to_field = "bob@gmail.com;alice@yahoo.com;carol@outlook.com";
+    DomainGroup groups[2] = {0};
+    int count = group_recipients_by_domain(to_field, groups, 2);
+    TEST_ASSERT_EQUAL_INT(2, count);
+
+    int found_gmail = 0, found_yahoo = 0, found_outlook = 0;
+    for (int i = 0; i < count; i++) {
+        if (strcmp(groups[i].domain, "gmail.com") == 0) found_gmail = 1;
+        if (strcmp(groups[i].domain, "yahoo.com") == 0) found_yahoo = 1;
+        if (strcmp(groups[i].domain, "outlook.com") == 0) found_outlook = 1;
+    }
+
+    TEST_ASSERT_TRUE(found_gmail || found_yahoo);
+    TEST_ASSERT_FALSE(found_outlook);
+}
+
+/**
+ * Tests that all recipients from the same domain are grouped together in a single entry.
+ */
+void test_group_recipients_by_domain_multiple_same_domain(void) {
+    const char *to_field = "a@x.com;b@x.com;c@x.com";
+    DomainGroup groups[1] = {0};
+
+    int count = group_recipients_by_domain(to_field, groups, 1);
+    TEST_ASSERT_EQUAL_INT(1, count);
+    TEST_ASSERT_EQUAL_STRING("x.com", groups[0].domain);
+    TEST_ASSERT_EQUAL_STRING("a@x.com;b@x.com;c@x.com", groups[0].recipients);
+}

```

### .gitignore
```diff
@@ -8,4 +8,5 @@ tests/Unity
 **/*.gcda
 **/*.gcno
 **/*.gcov
-tests/test_spamrock
\ No newline at end of file
+tests/test_spamrock
+tests/test_utils
\ No newline at end of file

```

### ChangeLog
```diff
@@ -1,3 +1,6 @@
+0.13.0 [2025-07-17]
+  - Fix(spamrock.c#send_shiva_notif): Split notif by domains
+
 0.12.1 [2025-06-18]
   - Fix(spamrock.c): init fail to
 

```
