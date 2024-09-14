# Utiliser une image Python comme base
FROM python:3.10-slim

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers du projet dans le container
COPY . /app
COPY requirements.txt /app

# Installer les dépendances
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Exposer le port sur lequel l'application va tourner
EXPOSE 8000

# Lancer l'application
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
