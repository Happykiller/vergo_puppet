## Titre initiale
MR 48 – [MR][SP][Shiva] Unicité Message_id et Impact du passage de champs "To" des demandes d'un mail à une liste de mail séparé par un ";"

## Description initiale
* Lever de l'unicité du message id
* Impact du passage de champs "To" des demandes d'un mail à une liste de mail séparé par un ";"

[\[SP\]\[MultiDest\]\[Shiva\] Lever de l'unicité du message id](https://thomyris.atlassian.net/browse/LF-4309)

[\[SP\]\[MultiDest\]\[Shiva\] Impact du passage de champs "To" des demandes d'un mail à une liste de mail séparé par un ";"](https://thomyris.atlassian.net/browse/LF-4310)

## Fichiers modifiés
- M app/Policies/EmailPolicy.php
- M app/Rest/Resources/EmailsResource.php
- M app/Services/PermissionQueries/Eloquent/Email.php
- A database/migrations/tenant/2025_06_10_131330_drop_unique_index_from_message_id_in_emails_table.php
- M tests/Feature/Apis/EmailsTest.php
- A tests/Unit/Policies/EmailPolicyTest.php
- A tests/Unit/Rest/Resources/EmailsResourceTest.php
- A tests/Unit/Services/PermissionQueries/Eloquent/EmailTest.php

## Diff complet

### app/Policies/EmailPolicy.php
```diff
@@ -26,9 +26,17 @@ class EmailPolicy
 {
     use HandlesPermissionPolicy;
 
-    protected function isOwn(User $user, Model $model): bool
+    public function isOwn(User $user, Model $model): bool
     {
-        return $model->to === $user->email;
+        if (empty($user->email) || empty($model->to)) {
+            return false;
+        }
+
+        $toList = explode(';', $model->to);
+
+        $cleanedToList = array_map('trim', $toList);
+
+        return in_array($user->email, $cleanedToList, true);
     }
 
     public function create(?User $user): bool

```

### app/Rest/Resources/EmailsResource.php
```diff
@@ -10,7 +10,6 @@
 
 use App\Models\Email;
 use App\Models\EmailStatus;
-use App\Rules\UniqueMessageId;
 use App\Rest\Resource as RestResource;
 use App\Services\PermissionQueries\Eloquent\Email as EmailEloquent;
 
@@ -113,7 +112,7 @@ public function rules(RestRequest $request): array {
 
     public function createRules(RestRequest $request): array {
         return [
-            'message_id' => ['required', new UniqueMessageId(self::$model)],
+            'message_id' => ['required'],
             'origin' => ['required'],
             'from' => ['required'],
             'to' => ['required'],

```

### app/Services/PermissionQueries/Eloquent/Email.php
```diff
@@ -12,8 +12,17 @@ public function implementQuery(Builder $query)
     {
         if ($this->auth->can('view_emails')) {
             return $query;
-        } elseif ($this->auth->can('view_own_emails')) {
-            return $query->where('emails.to', $this->auth->email);
+        }
+
+        if ($this->auth->can('view_own_emails')) {
+            return $query->where(function (Builder $subQuery) {
+                $userEmail = $this->auth->email;
+
+                $subQuery->where('emails.to', $userEmail)
+                    ->orWhere('emails.to', 'like', $userEmail . ';%')
+                    ->orWhere('emails.to', 'like', '%;' . $userEmail . ';%')
+                    ->orWhere('emails.to', 'like', '%;' . $userEmail);
+            });
         }
 
         return $query->whereRaw('0 = 1');

```

### database/migrations/tenant/2025_06_10_131330_drop_unique_index_from_message_id_in_emails_table.php
```diff
@@ -0,0 +1,22 @@
+<?php
+
+use Illuminate\Support\Facades\Schema;
+use Illuminate\Database\Schema\Blueprint;
+use Illuminate\Database\Migrations\Migration;
+
+return new class extends Migration
+{
+    public function up(): void
+    {
+        Schema::table('emails', function (Blueprint $table) {
+            $table->dropUnique('emails_message_id_unique');
+        });
+    }
+
+    public function down(): void
+    {
+        Schema::table('emails', function (Blueprint $table) {
+            $table->unique('message_id');
+        });
+    }
+};

```

### tests/Feature/Apis/EmailsTest.php
```diff
@@ -28,7 +28,9 @@ protected function setUp(): void
         $this->setupDatabase();
 
         Event::fake();
-        $user = User::factory()->create();
+        $user = User::factory()->create([
+            'email' => 'current-user@example.com'
+        ]);
         $user->assignRole('User');
         Passport::actingAs($user);
 
@@ -63,6 +65,51 @@ public function test_get_emails(): void
             "meta"
         ]);
     }
+
+    #[Test]
+    public function it_retrieves_only_emails_addressed_to_the_authenticated_user(): void
+    {
+        // ARRANGE
+        $emailForUser = Email::factory()->create([
+            'status_id' => Email::STATUS_NEUTRAL,
+            'message_id' => '42',
+            'to' => 'other@test.com;current-user@example.com',
+            'subject' => 'Email FOR me'
+        ]);
+
+        $emailForOthers = Email::factory()->create([
+            'status_id' => Email::STATUS_NEUTRAL,
+            'message_id' => '42',
+            'to' => 'other@test.com;another@test.com',
+            'subject' => 'Email NOT for me'
+        ]);
+
+        $anotherEmailForUser = Email::factory()->create([
+            'status_id' => Email::STATUS_NEUTRAL,
+            'message_id' => '42',
+            'to' => 'current-user@example.com',
+            'subject' => 'Another email FOR me'
+        ]);
+
+        // ACT
+        $response = $this->postJson($this->getApiUrl('emails/search/'));
+
+        // ASSERT
+        $response->assertStatus(200);
+
+        $response->assertJsonCount(2, 'data');
+        $response->assertJson(['total' => 2]);
+
+        $response->assertJsonFragment(['id' => $emailForUser->id]);
+        $response->assertJsonFragment(['subject' => 'Email FOR me']);
+
+        $response->assertJsonFragment(['id' => $anotherEmailForUser->id]);
+        $response->assertJsonFragment(['subject' => 'Another email FOR me']);
+
+        $response->assertJsonMissing(['id' => $emailForOthers->id]);
+        $response->assertJsonMissing(['subject' => 'Email NOT for me']);
+    }
+
     #[Test]
     public function user_can_update_email_status(): void
     {

```

### tests/Unit/Policies/EmailPolicyTest.php
```diff
@@ -0,0 +1,151 @@
+<?php
+
+namespace Tests\Unit\Policies;
+
+use Mockery;
+use PHPUnit\Framework\Attributes\Test;
+use PHPUnit\Framework\Attributes\DataProvider;
+
+use Tests\TestCase;
+use App\Models\User;
+use App\Models\Email;
+use App\Policies\EmailPolicy;
+
+class EmailPolicyTest extends TestCase
+{
+    protected function tearDown(): void
+    {
+        Mockery::close();
+        parent::tearDown();
+    }
+
+    #[Test]
+    public function is_own_returns_true_when_user_email_matches_model_to_field(): void
+    {
+        // ARRANGE
+        $user = new User();
+        $user->email = 'owner@example.com';
+
+        $model = new Email();
+        $model->to = 'owner@example.com';
+
+        $policy = new EmailPolicy();
+
+        // ACT & ASSERT
+        $this->assertTrue($policy->isOwn($user, $model));
+    }
+
+    #[Test]
+    public function is_own_returns_false_when_user_email_does_not_match_model_to_field(): void
+    {
+        // ARRANGE
+        $user = new User();
+        $user->email = 'not.the.owner@example.com';
+
+        $model = new Email();
+        $model->to = 'owner@example.com';
+
+        $policy = new EmailPolicy();
+
+        // ACT & ASSERT
+        $this->assertFalse($policy->isOwn($user, $model));
+    }
+
+    #[Test]
+    public function create_returns_true_for_a_guest_user(): void
+    {
+        // ARRANGE
+        $policy = new EmailPolicy();
+
+        // ACT & ASSERT
+        $this->assertTrue($policy->create(null));
+    }
+
+    #[Test]
+    public function create_delegates_to_user_can_method_for_an_authenticated_user(): void
+    {
+        // ARRANGE
+        $user = Mockery::mock(User::class);
+
+        $user->shouldReceive('can')
+            ->with('create_emails')
+            ->once()
+            ->andReturn(true);
+
+        $policy = new EmailPolicy();
+
+        // ACT & ASSERT
+        $this->assertTrue($policy->create($user));
+
+        // ARRANGE
+        $userWithoutPermission = Mockery::mock(User::class);
+        $userWithoutPermission->shouldReceive('can')
+            ->with('create_emails')
+            ->once()
+            ->andReturn(false);
+
+        // ACT & ASSERT
+        $this->assertFalse($policy->create($userWithoutPermission));
+    }
+
+    #[Test]
+    #[DataProvider('permissionPolicyDataProvider')]
+    public function standard_policy_methods_return_false_without_any_permission(string $policyMethod, string $globalPermission, string $ownPermission): void
+    {
+        // ARRANGE
+        $user = Mockery::mock(User::class);
+        $model = Mockery::mock(Email::class);
+        $policy = new EmailPolicy();
+
+        $user->shouldReceive('can')->with($globalPermission)->once()->andReturn(false);
+        $user->shouldReceive('can')->with($ownPermission)->once()->andReturn(false);
+
+        // ACT & ASSERT
+        $this->assertFalse($policy->{$policyMethod}($user, $model));
+    }
+
+
+    public static function permissionPolicyDataProvider(): array
+    {
+        return [
+            'view'        => ['view', 'view_emails', 'view_own_emails'],
+            'update'      => ['update', 'update_emails', 'update_own_emails'],
+            'delete'      => ['delete', 'delete_emails', 'delete_own_emails'],
+            'restore'     => ['restore', 'restore_emails', 'restore_own_emails'],
+            'forceDelete' => ['forceDelete', 'force_delete_emails', 'force_delete_own_emails'],
+        ];
+    }
+
+    #[Test]
+    #[DataProvider('emailListDataProvider')]
+    public function is_own_handles_semicolon_separated_email_lists(?string $userEmail, ?string $toList, bool $expectedResult): void
+    {
+        // ARRANGE
+        $user = new User();
+        $user->email = $userEmail;
+
+        $model = new Email();
+        $model->to = $toList;
+
+        $policy = new EmailPolicy();
+
+        // ACT & ASSERT
+        $this->assertSame($expectedResult, $policy->isOwn($user, $model));
+    }
+
+    public static function emailListDataProvider(): array
+    {
+        $userEmail = 'user@example.com';
+        return [
+            'user email is in the middle of the list'     => [$userEmail, 'other@test.com;user@example.com;another@test.com', true],
+            'user email is at the start of the list'      => [$userEmail, 'user@example.com;another@test.com', true],
+            'user email is at the end of the list'        => [$userEmail, 'another@test.com;user@example.com', true],
+            'list contains extra whitespace'              => [$userEmail, ' another@test.com ; user@example.com ; final@test.com ', true],
+            'user email is not in the list'               => [$userEmail, 'other@test.com;another@test.com', false],
+            'user email is a substring of another email'  => ['owner@test.com', 'super-owner@test.com;other@test.com', false],
+            'to field is null'                            => [$userEmail, null, false],
+            'to field is an empty string'                 => [$userEmail, '', false],
+            'user email is null'                          => [null, 'some@email.com;another@email.com', false],
+        ];
+    }
+}

```

### tests/Unit/Rest/Resources/EmailsResourceTest.php
```diff
@@ -0,0 +1,41 @@
+<?php
+
+namespace Rest\Resources;
+
+use Mockery;
+use App\Rules\UniqueMessageId;
+use PHPUnit\Framework\Attributes\Test;
+use Lomkit\Rest\Http\Requests\RestRequest;
+
+use Tests\TestCase;
+use App\Rest\Resources\EmailsResource;
+
+class EmailsResourceTest extends TestCase
+{
+    protected function tearDown(): void
+    {
+        Mockery::close();
+        parent::tearDown();
+    }
+
+    #[Test]
+    public function it_returns_expected_structure()
+    {
+        // Arrange
+        $resource = new EmailsResource();
+        /** @var RestRequest $request */
+        $request = Mockery::mock(RestRequest::class);
+
+        // Act
+        $rules = $resource->createRules($request);
+
+        // Assert
+        $this->assertArrayHasKey('message_id', $rules);
+        $messageIdRules = collect($rules['message_id']);
+
+        $this->assertFalse(
+            $messageIdRules->contains(fn ($rule) => $rule instanceof UniqueMessageId),
+            'The UniqueMessageId rule is still present in createRules for message_id.'
+        );
+    }
+}

```

### tests/Unit/Services/PermissionQueries/Eloquent/EmailTest.php
```diff
@@ -0,0 +1,126 @@
+<?php
+
+namespace Tests\Unit\Services\PermissionQueries\Eloquent;
+
+use Mockery;
+use ReflectionException;
+use Mockery\MockInterface;
+use PHPUnit\Framework\Attributes\Test;
+use Illuminate\Database\Eloquent\Builder;
+
+use Tests\TestCase;
+use App\Models\User;
+use App\Services\PermissionQueries\Eloquent\Email as EmailPermissionQuery;
+
+class EmailTest extends TestCase
+{
+    private MockInterface|Builder $queryBuilderMock;
+    private MockInterface|User $userMock;
+    private EmailPermissionQuery $permissionQuery;
+
+    protected function setUp(): void
+    {
+        parent::setUp();
+
+        $this->queryBuilderMock = Mockery::mock(Builder::class);
+        $this->userMock = Mockery::mock(User::class);
+
+        $this->permissionQuery = new EmailPermissionQuery();
+
+        $this->setProtectedProperty($this->permissionQuery, 'auth', $this->userMock);
+    }
+
+    protected function tearDown(): void
+    {
+        Mockery::close();
+        parent::tearDown();
+    }
+
+    #[Test]
+    public function it_returns_the_query_unmodified_when_user_can_view_all_emails(): void
+    {
+        // ARRANGE
+        $this->userMock->shouldReceive('can')
+            ->with('view_emails')
+            ->once()
+            ->andReturn(true);
+
+        $this->queryBuilderMock->shouldNotReceive('where');
+        $this->queryBuilderMock->shouldNotReceive('whereRaw');
+        $this->queryBuilderMock->shouldNotReceive('orWhere');
+
+        // ACT
+        $result = $this->permissionQuery->implementQuery($this->queryBuilderMock);
+
+        // ASSERT
+        $this->assertSame($this->queryBuilderMock, $result);
+    }
+
+    #[Test]
+    public function it_applies_ownership_filter_when_user_can_only_view_own_emails(): void
+    {
+        // ARRANGE
+        $userEmail = 'owner@example.com';
+
+        $this->userMock->shouldReceive('getAttribute')
+            ->with('email')
+            ->andReturn($userEmail);
+
+        $this->userMock->shouldReceive('can')->with('view_emails')->once()->andReturn(false);
+        $this->userMock->shouldReceive('can')->with('view_own_emails')->once()->andReturn(true);
+
+        $this->queryBuilderMock->shouldReceive('where')
+            ->once()
+            ->with(Mockery::on(function ($closure) use ($userEmail) {
+                $subQueryMock = Mockery::mock(Builder::class);
+                $subQueryMock->shouldReceive('where')->with('emails.to', $userEmail)->once()->andReturnSelf();
+                $subQueryMock->shouldReceive('orWhere')->with('emails.to', 'like', $userEmail . ';%')->once()->andReturnSelf();
+                $subQueryMock->shouldReceive('orWhere')->with('emails.to', 'like', '%;' . $userEmail . ';%')->once()->andReturnSelf();
+                $subQueryMock->shouldReceive('orWhere')->with('emails.to', 'like', '%;' . $userEmail)->once()->andReturnSelf();
+                $closure($subQueryMock);
+                return true;
+            }))
+            ->andReturnSelf();
+
+        // ACT
+        $result = $this->permissionQuery->implementQuery($this->queryBuilderMock);
+
+        // ASSERT
+        $this->assertSame($this->queryBuilderMock, $result);
+    }
+
+    #[Test]
+    public function it_applies_a_blocking_filter_when_user_has_no_view_permissions(): void
+    {
+        // ARRANGE
+        $this->userMock->shouldReceive('can')->with('view_emails')->once()->andReturn(false);
+        $this->userMock->shouldReceive('can')->with('view_own_emails')->once()->andReturn(false);
+
+        $this->queryBuilderMock->shouldReceive('whereRaw')
+            ->with('0 = 1')
+            ->once()
+            ->andReturnSelf();
+
+        $this->queryBuilderMock->shouldNotReceive('where');
+
+        // ACT
+        $result = $this->permissionQuery->implementQuery($this->queryBuilderMock);
+
+        // ASSERT
+        $this->assertSame($this->queryBuilderMock, $result);
+    }
+
+    /**
+     * @param object $object
+     * @param string $property
+     * @param mixed $value
+     * @throws ReflectionException
+     */
+    private function setProtectedProperty(object $object, string $property, $value): void
+    {
+        $reflection = new \ReflectionClass($object);
+        $reflectionProperty = $reflection->getProperty($property);
+        $reflectionProperty->setAccessible(true);
+        $reflectionProperty->setValue($object, $value);
+    }
+}

```
