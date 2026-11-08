# LLMaaltijd ML proof of concept: Python-service

Proof of concept van Kane de Nooijer voor BP5 AI Food Prep (LLMaaltijd).

Deze service schat met machine learning de **NOVA-groep** van een product (hoe sterk het bewerkt is, van 1
tot 4) als Open Food Facts die niet kent. De Laravel-app
[ml-laravel](https://github.com/KaneDeNooijer/ml-laravel) toont die schatting na het scannen van een product.

- **De analyse** achter alle keuzes staat in [`docs/analyse.md`](docs/analyse.md).
- **Het ML-component:** scikit-learn (k-Nearest Neighbours op TF-IDF-kenmerken), in een FastAPI-service.
- **Design patterns:** Strategy (uitwisselbare modellen), Adapter (Open Food Facts en Laravel), Pipeline.

---

## Voor de docent: opzetten en testen

### 1. Wat je nodig hebt

- [Docker Desktop](https://www.docker.com/products/docker-desktop/), gestart.
- Git.
- Internet: de service haalt productgegevens live op bij Open Food Facts.

Python hoef je niet zelf te installeren; dat zit in de container.

### 2. Beide repositories ophalen

De proof of concept bestaat uit twee repositories. Zet ze naast elkaar:

```sh
git clone https://github.com/KaneDeNooijer/ml-python.git
git clone https://github.com/KaneDeNooijer/ml-laravel.git
```

### 3. De Python-service starten

```sh
cd ml-python
docker compose up --build
```

De eerste keer duurt dit een paar minuten. De service is klaar als je `Application startup complete` ziet.
Laat dit terminalvenster open.

Controleer dat hij draait: <http://localhost:8101/health> moet `{"status":"ok"}` tonen.

### 4. Het ML-deel testen in de browser

Open <http://localhost:8101/docs>. Dit is de interactieve API-documentatie: klap een endpoint open, klik
**Try it out** en dan **Execute**.

| Wat je test | Endpoint | Wat je hoort te zien |
| --- | --- | --- |
| **Een schatting door het model** | `GET /products/5400119506922` (yoghurt drink aardbei) | `"estimated": true`, `"group": 4`, het model (`k_nearest_neighbours`), de zekerheid, de redenen (kenmerken) en op hoeveel producten het getraind is |
| **Een bekende groep** | `GET /products/5410146416866` | `"estimated": false`: Open Food Facts kent de groep al, dus het model doet niets |
| **De vergelijking van de modellen** | `GET /evaluation?category=en:yogurts` | Nauwkeurigheid van alle vier modellen met 5-voudige cross-validatie, beste eerst. Probeer ook `en:breakfast-cereals` of `en:cheeses`. |
| **Een onbekend product** | `GET /products/00000000` | Status 404 |

De cijfers van `/evaluation` zijn dezelfde metingen als in hoofdstuk 8 van de analyse. Kleine verschillen kunnen
voorkomen, omdat Open Food Facts dagelijks wordt bijgewerkt.

### 5. De automatische tests draaien

Terwijl de service draait, in een tweede terminal in de map `ml-python`:

```sh
docker compose exec app python -m pytest
```

Er horen **11 tests** te slagen. Ze gebruiken een nagebootste Open Food Facts, dus ze hebben geen internet nodig.
Ze testen:
- dat een bekende groep ongewijzigd blijft;
- dat een ontbrekende groep wordt geschat uit dezelfde categorie;
- dat de groep onbekend blijft bij te weinig vergelijkbare producten;
- de foutafhandeling (404, 422, 502);
- de evaluatie tegen de baseline;
- dat elk van de vier modellen uitwisselbaar is (Strategy).

### 6. Samen met Laravel testen

Start daarna de Laravel-app volgens de README van
[ml-laravel](https://github.com/KaneDeNooijer/ml-laravel). Die verwacht deze service op poort 8101.

### 7. Stoppen

Druk `Ctrl+C` in het terminalvenster, of:

```sh
docker compose down
```

---

## Waar zit wat?

| Bestand | Inhoud | In de analyse |
| --- | --- | --- |
| `src/poc_ml/nova.py` | De kenmerken (tokens) en de vier uitwisselbare modellen: baseline, Naive Bayes, kNN, Logistic Regression | H6, H7.1 Strategy |
| `src/poc_ml/service.py` | Kiest de trainingsdata per categorie, schat de groep, evalueert de modellen | H6.1, H8.3 |
| `src/poc_ml/open_food_facts.py` | Haalt producten op bij Open Food Facts | H7.2 Adapter |
| `src/poc_ml/config.py` | Alle drempels en het gekozen model (`CHOSEN_MODEL`) | H9 |
| `src/poc_ml/api.py` | De HTTP-routes | H7.3 Microservice |
| `tests/test_api.py` | De automatische tests | |

## Endpoints

| Methode | Pad | Doel |
| --- | --- | --- |
| GET | `/health` | Draait de service? |
| GET | `/products/{barcode}` | Product met NOVA-groep; geschat met k-Nearest Neighbours als die ontbreekt |
| GET | `/evaluation?category=en:yogurts` | 5-voudige cross-validatie van alle modellen tegen de baseline |

## Zonder Docker (optioneel)

Vereist Python 3.13 en [uv](https://docs.astral.sh/uv/):

```sh
uv sync --group dev
uv run uvicorn poc_ml.api:app --port 8101 --reload
uv run pytest && uv run mypy && uv run ruff check .
```

## Problemen oplossen

| Probleem | Oplossing |
| --- | --- |
| `port is already allocated` | Poort 8101 is bezet. Start met een andere poort: `APP_PORT=8111 docker compose up`. Laravel moet dan `PYTHON_API_URL=http://host.docker.internal:8111` krijgen. |
| Status 502 met "Open Food Facts answered with status 429" | Open Food Facts beperkt het aantal aanvragen per netwerk. Wacht een paar minuten. |
| `"group": null` bij een product | Er waren te weinig vergelijkbare producten met een bekende groep (minimaal 20) om betrouwbaar te schatten. Dat is bedoeld gedrag. |
