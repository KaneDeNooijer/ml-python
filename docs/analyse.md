# Analyse: machine learning-componenten voor LLMaaltijd

Persoonlijke analyse van Kane de Nooijer, als onderdeel van het gezamenlijke adviesrapport.
De proof of concept staat in twee repositories: `ml-python` (het ML-onderdeel) en `ml-laravel` (de app die
het gebruikt).

## Inhoud

1. Aanleiding en onderzoeksvraag
2. Soorten machine learning en hun plek in LLMaaltijd
3. Eisen vanuit LLMaaltijd
4. Selectiecriteria
5. Analyse van de ML-componenten
6. Analyse van de algoritmes binnen het gekozen component
7. Design patterns
8. Vergelijkende analyse
9. Conclusie en advies
10. Beperkingen en vervolg
11. Bronnen

---

## 1. Aanleiding en onderzoeksvraag

LLMaaltijd is een webapplicatie voor meal prep. Gebruikers zoeken en scannen producten, plannen maaltijden per
week en krijgen per product een Prep-score die aangeeft hoe gezond het is. Alle productgegevens komen uit
Open Food Facts, een open database die door gebruikers wordt aangevuld. Dat maakt de data breed beschikbaar en
gratis, maar ook onvolledig.

Een belangrijk onderdeel van de Prep-score is de **NOVA-groep**: een internationale indeling van voedsel naar
mate van bewerking, van groep 1 (onbewerkt of minimaal bewerkt, zoals melk of groente) tot groep 4
(ultrabewerkt, zoals frisdrank of koek) (Monteiro et al., 2019). Bij veel producten ontbreekt die groep. Van
de 600 Nederlandse producten die ik voor deze analyse in zes categorieën heb opgehaald, had maar 379 een
NOVA-groep. Ruim een derde mist hem dus.

Zonder NOVA-groep is de Prep-score onvollediger en kan de app de gebruiker niet vertellen hoe bewerkt een
product is. Handmatig aanvullen is onmogelijk bij miljoenen producten. Dit is een klassiek geval waarin machine
learning kan helpen: er is een grote hoeveelheid voorbeelden mét antwoord (producten met een bekende groep)
waaruit een model het patroon kan leren, om dat daarna toe te passen op producten zonder antwoord.

Mijn onderzoeksvraag is:

> **Welk bestaand machine learning-component is het meest geschikt om in LLMaaltijd ontbrekende NOVA-groepen
> te schatten, en hoe integreer je dat component met de Laravel-applicatie?**

Daarbij horen deze deelvragen:

1. Welke soorten machine learning bestaan er, en welke passen bij dit probleem en bij LLMaaltijd als geheel?
2. Aan welke eisen moet een oplossing voldoen, en welke selectiecriteria volgen daaruit?
3. Welke bestaande ML-componenten komen in aanmerking, en wat zijn hun technische eigenschappen, voor- en
   nadelen en toepassingen?
4. Welk algoritme presteert binnen het gekozen component het best op echte Nederlandse productdata?
5. Welke design patterns maken de integratie onderhoudbaar en testbaar?

Ik beantwoord deze vragen met literatuur en documentatie, en meet waar dat kan zelf. De metingen in
hoofdstuk 6 komen uit de proof of concept en zijn met het endpoint `/evaluation` te herhalen.

---

## 2. Soorten machine learning en hun plek in LLMaaltijd

Machine learning is het vakgebied waarin een computer een taak leert uit voorbeelden, in plaats van dat een
programmeur elke regel uitschrijft (Géron, 2022). Er zijn vier hoofdsoorten, die verschillen in wat voor data
ze nodig hebben en wat voor vraag ze beantwoorden. Daarnaast onderscheid ik de soorten bestaande componenten
waarmee je die technieken in een applicatie gebruikt.

### 2.1 Supervised learning

Bij supervised learning leert een model van voorbeelden waarvan het juiste antwoord (het label) bekend is. Het
model zoekt het verband tussen de kenmerken van een voorbeeld en het label, en voorspelt daarna het label van
nieuwe voorbeelden. Er zijn twee varianten. Bij **classificatie** is het antwoord een categorie, zoals "spam"
of "geen spam". Bij **regressie** is het antwoord een getal, zoals een huizenprijs.

In LLMaaltijd past supervised learning op twee plekken. Het schatten van de NOVA-groep is **classificatie**:
het antwoord is één van vier groepen, en er zijn duizenden producten met een bekende groep om van te leren.
Het schatten van ontbrekende voedingswaarden, zoals het suikergehalte per 100 gram, is **regressie**: het
antwoord is een getal, en producten met bekende voedingswaarden dienen als voorbeelden.

### 2.2 Unsupervised learning

Bij unsupervised learning zijn er geen labels. Het model zoekt zelf structuur in de data, bijvoorbeeld groepen
van gelijksoortige voorbeelden (**clustering**), of een compactere weergave met minder dimensies
(**dimensiereductie**).

In LLMaaltijd past dit bij **aanbevelingen**. Uit welke producten gebruikers bekijken, scannen en plannen kan
een model verborgen voorkeuren afleiden zonder dat iemand die voorkeuren benoemt. Matrixfactorisatie, een
vorm van dimensiereductie, is daar de bekende techniek voor (Koren et al., 2009). Ook het vinden van
vergelijkbare producten voor een gezonder alternatief is in de kern unsupervised: het gaat om gelijkenis,
niet om een vooraf bekend antwoord.

### 2.3 Semi-supervised learning

Semi-supervised learning combineert een kleine hoeveelheid gelabelde data met veel ongelabelde data. Een
bekende aanpak is **self-training**: het model labelt de ongelabelde voorbeelden waarvan het zeker is, en
leert daar vervolgens ook van (Yarowsky, 1995).

Dit past precies bij de NOVA-groep. De producten zónder groep zijn niet alleen het probleem, maar ook bruikbare
ongelabelde data. Voor de proof of concept kies ik eerst voor gewone supervised classificatie, omdat ik dan
zuiver kan meten wat een model met alleen gelabelde data kan. Self-training noem ik als vervolgstap in
hoofdstuk 10.

### 2.4 Reinforcement learning

Bij reinforcement learning leert een agent door acties uit te voeren in een omgeving en daarvoor beloning of
straf te krijgen. Het wordt gebruikt voor spellen, robotica en het optimaliseren van beslissingen over tijd.

Voor LLMaaltijd past dit niet. Er is geen omgeving waarin een model zelf handelt en direct beloning krijgt, en
experimenteren met wat gebruikers te zien krijgen om een beloningssignaal op te bouwen, weegt niet op tegen de
eenvoudigere technieken hierboven.

### 2.5 Klassieke machine learning, deep learning en taalmodellen

Binnen deze soorten maak ik nog onderscheid naar het type model. **Klassieke machine learning** (zoals
lineaire modellen, beslisbomen en nearest neighbours) werkt goed met gestructureerde data en kleine datasets,
is snel en vaak uitlegbaar. **Deep learning** (neurale netwerken met veel lagen) is sterk bij ongestructureerde
data zoals beeld, spraak en lange teksten, maar heeft veel data en rekenkracht nodig en is moeilijker uit te
leggen. **Grote taalmodellen** (LLM's) zijn vooraf getrainde deep learning-modellen die via een tekstprompt
taken kunnen uitvoeren zonder eigen training ("zero-shot"), maar die betaald, traag en niet altijd
reproduceerbaar zijn.

Voor de NOVA-groep heb ik per categorie zo'n 60 gelabelde Nederlandse producten, en elk product bestaat uit
korte gestructureerde kenmerken: woorden uit de naam, categorieën, ingrediënten en het aantal toevoegingen.
Dat wijst sterk richting klassieke machine learning.

### 2.6 Soorten ML-componenten

Naast de soort machine learning moet ik kiezen *waarmee* ik die toepas. Bestaande ML-componenten vallen in
vijf groepen:

| Soort component | Voorbeelden | Kenmerk |
| --- | --- | --- |
| **Bibliotheek voor klassieke ML** | scikit-learn (Python), Rubix ML en PHP-ML (PHP) | Je traint en draait modellen in je eigen code. |
| **Deep learning-framework** | TensorFlow/Keras, PyTorch | Voor neurale netwerken; zwaar, maar zeer flexibel. |
| **Model-runtime** | ONNX Runtime | Draait een elders getraind model in een andere taal of omgeving. |
| **Beheerde clouddienst** | Google Vertex AI, AWS SageMaker, Azure Machine Learning | De cloudaanbieder traint en host het model; je betaalt per gebruik. |
| **Taalmodel-API** | OpenAI, Anthropic, Mistral | Een vooraf getraind taalmodel beantwoordt een prompt; betaald per token. |

### 2.7 Vertaling naar mijn context

[Vul hier je uitstroomprofiel in.] Als developer van LLMaaltijd is mijn taak niet om nieuwe ML-algoritmes te
ontwikkelen, maar om een bestaand ML-component zo te kiezen en te integreren dat het betrouwbaar, uitlegbaar
en onderhoudbaar in een webapplicatie werkt. Daarom kijk ik in deze analyse op twee niveaus. Eerst vergelijk ik
de **componenten**: met welke software train en draai ik het model, en hoe praat Laravel ermee? Daarna
vergelijk ik binnen het gekozen component de **algoritmes**: welke techniek schat de NOVA-groep het best?

---

## 3. Eisen vanuit LLMaaltijd

De selectiecriteria volgen uit wat de applicatie en haar gebruikers nodig hebben. Ik heb de eisen opgesteld
vanuit de bestaande architectuur (een Laravel-applicatie met MongoDB) en vanuit hoe de NOVA-groep in de
interface wordt getoond.

**Functionele eisen**

1. De oplossing schat de NOVA-groep van een product als Open Food Facts die niet kent.
2. Een bekende NOVA-groep wordt nooit overschreven door een schatting.
3. De app laat zien dát een groep geschat is, hoe zeker het model is en waarop de schatting is gebaseerd.
4. Als er te weinig vergelijkbare producten zijn om betrouwbaar te schatten, blijft de groep onbekend.
5. De kwaliteit van de schattingen is meetbaar en te vergelijken met een eenvoudige baseline.

**Niet-functionele eisen**

1. **Integratie:** de oplossing is vanuit Laravel aan te roepen.
2. **Snelheid:** een schatting is binnen een seconde klaar, omdat de gebruiker in de winkel op het resultaat
   van een scan wacht.
3. **Kosten:** de oplossing heeft geen doorlopende kosten per gebruik; het is een studentproject zonder budget.
4. **Onderhoud:** het component wordt actief onderhouden en is goed gedocumenteerd.
5. **Testbaarheid:** het ML-onderdeel is automatisch te testen zonder netwerk of externe diensten.
6. **Privacy:** er gaan geen persoonsgegevens naar derden. Voor de NOVA-groep is alleen openbare productdata
   nodig, maar dezelfde integratie wordt later ook gebruikt voor aanbevelingen op basis van gebruikersgedrag.

---

## 4. Selectiecriteria

Ik beoordeel in twee stappen. Eerst vallen componenten af die niet aan een harde eis voldoen
(**knock-out-criteria**). De overgebleven componenten krijgen een gewogen score op de **wegingscriteria**.
Voor de algoritmes binnen het gekozen component gebruik ik een eigen set wegingscriteria, omdat daar andere
eigenschappen tellen.

### 4.1 Knock-out-criteria voor componenten

| Code | Criterium | Waarom hard |
| --- | --- | --- |
| K1 | Aan te roepen vanuit Laravel | Zonder integratie is het component voor LLMaaltijd onbruikbaar. |
| K2 | Geen doorlopende kosten per gebruik | Er is geen budget; kosten per scan zijn niet te dragen bij groei. |
| K3 | Actief onderhouden | Een onbeheerd component krijgt geen beveiligingsupdates en loopt achter op nieuwe PHP- of Python-versies. |
| K4 | Werkt met ~60 gelabelde voorbeelden per categorie | Meer gelabelde Nederlandse data is er per categorie niet. |

### 4.2 Wegingscriteria voor componenten

| Criterium | Gewicht | Onderbouwing | Hoe gescoord |
| --- | --- | --- | --- |
| **Beschikbare algoritmes en evaluatietools** | 20% | Ik wil meerdere algoritmes kunnen vergelijken en de kwaliteit kunnen meten (eis F5). | Aantal geschikte algoritmes; aanwezigheid van cross-validatie en metrieken. |
| **Volwassenheid en documentatie** | 20% | Een volwassen component met grote gemeenschap heeft minder fouten en sneller antwoorden op vragen (eis NF4). | Gebruik in de praktijk, documentatie, releasefrequentie. |
| **Integratie-inspanning met Laravel** | 20% | Hoe minder losse onderdelen, hoe minder er stuk kan (eis NF1). | Wat er extra nodig is naast de bestaande Laravel-app. |
| **Onderhoudbaarheid en testbaarheid** | 15% | Het ML-onderdeel moet los te testen en te vervangen zijn (eis NF5). | Scheiding van verantwoordelijkheden, testmogelijkheden. |
| **Aansluiting op kennis in het team** | 15% | Het team moet het later kunnen onderhouden en uitleggen. | Bekendheid uit de lesstof en eerdere projecten. |
| **Prestaties** | 10% | Een schatting moet binnen een seconde klaar zijn (eis NF2); alle kandidaten halen dat ruim, dus het weegt minder zwaar. | Verwachte responstijd. |

### 4.3 Wegingscriteria voor algoritmes

| Criterium | Gewicht | Onderbouwing | Hoe gemeten |
| --- | --- | --- | --- |
| **Nauwkeurigheid** | 30% | Een schatting heeft alleen waarde als ze vaker klopt dan simpelweg de meest voorkomende groep invullen. | Macro-F1 en accuracy met 5-voudige cross-validatie, vergeleken met een baseline. |
| **Uitlegbaarheid** | 25% | De gebruiker moet kunnen zien waarop een schatting is gebaseerd (eis F3). | Kan het model per schatting de doorslaggevende kenmerken noemen? |
| **Werkt met weinig data** | 15% | Er zijn per categorie maar ~60 gelabelde producten. | Gedrag bij kleine datasets, volgens literatuur en meting. |
| **Snelheid** | 15% | Het model traint op het moment van de scan (eis NF2). | Gemeten trainings- en voorspeltijd. |
| **Bruikbare zekerheid** | 10% | De app toont hoe zeker het model is (eis F3). | Geeft het model kansen die iets betekenen? |
| **Integratie** | 5% | Alle kandidaten zitten in hetzelfde component, dus dit onderscheidt nauwelijks. | Beschikbaar in scikit-learn. |

Ik meet nauwkeurigheid met **macro-F1** naast accuracy. Accuracy is het aandeel goede schattingen. Dat is
misleidend als de groepen ongelijk verdeeld zijn: bij koek is 70 van de 74 producten groep 4, dus een model dat
altijd "4" zegt heeft al 95% accuracy zonder iets te leren. Macro-F1 berekent de F1-score (het harmonisch
gemiddelde van precisie en recall) per groep en neemt daar het gemiddelde van. Een model scoort dan alleen
hoog als het ook de kleine groepen herkent.

---

## 5. Analyse van de ML-componenten

Per component beschrijf ik de werking, de technische specificaties, de voor- en nadelen en de mogelijke
toepassingen. De beoordeling volgt in hoofdstuk 8.

### 5.1 scikit-learn in een Python-service

**Beschrijving.** scikit-learn is de meest gebruikte Python-bibliotheek voor klassieke machine learning
(Pedregosa et al., 2011). Laravel kan geen Python draaien, dus het model draait in een aparte webservice die
Laravel via HTTP aanroept. Voor die service gebruik ik FastAPI, een Python-framework voor API's dat
invoer valideert met Pydantic en automatisch documentatie genereert.

**Technische specificaties.**

| Eigenschap | Waarde |
| --- | --- |
| Taal | Python 3.13 |
| Licentie | BSD-3 (open source, gratis, ook commercieel) |
| Algoritmes | Lineaire modellen, Naive Bayes, nearest neighbours, beslisbomen, random forests, gradient boosting, SVM's, clustering, dimensiereductie, semi-supervised (self-training) |
| Tekstverwerking | `TfidfVectorizer`, `CountVectorizer` |
| Evaluatie | `KFold`, `cross_val_score`, `f1_score`, confusion matrix, `GridSearchCV` |
| Integratie met Laravel | HTTP met JSON via een eigen FastAPI-service |
| Hosting | Eigen container naast de Laravel-app |

**Voordelen.**
- Zeer volwassen, met uitstekende documentatie en een grote gemeenschap; vrijwel elke ML-vraag is al eens
  beantwoord.
- Alle algoritmes delen dezelfde interface (`fit` en `predict`). Daardoor zijn modellen eenvoudig te
  vergelijken en te vervangen.
- Complete evaluatietools: cross-validatie en metrieken zitten in de bibliotheek.
- Python is de standaardtaal in de ML-wereld en sluit aan op de lesstof.
- De service scheidt ML-code van de webapplicatie, dus beide zijn los te testen, te deployen en te schalen.

**Nadelen.**
- Een extra service betekent een extra container, een netwerkverbinding en een extra punt dat kan uitvallen.
  Laravel moet daar netjes op reageren.
- Elke aanvraag kost een HTTP-hop van een paar milliseconden.
- Het team moet twee talen beheersen: PHP en Python.

**Mogelijke toepassingen.** Classificatie (NOVA-groep), regressie (ontbrekende voedingswaarden), aanbevelingen
met matrixfactorisatie, en gelijkenis zoeken voor alternatieven. Buiten LLMaaltijd wordt scikit-learn breed
gebruikt voor fraudedetectie, klantsegmentatie, churn-voorspelling en vraagvoorspelling.

### 5.2 Rubix ML in Laravel

**Beschrijving.** Rubix ML is een machine learning-bibliotheek in PHP. Het model draait in hetzelfde proces als
Laravel, dus er is geen aparte service nodig.

**Technische specificaties.**

| Eigenschap | Waarde |
| --- | --- |
| Taal | PHP |
| Licentie | MIT (open source, gratis) |
| Algoritmes | O.a. k-nearest neighbors, logistic regression, Naive Bayes, beslisbomen, random forest, neurale netwerken, clustering |
| Tekstverwerking | `WordCountVectorizer`, `TfIdfTransformer` |
| Evaluatie | Cross-validatie (`KFold`) en metrieken zoals accuracy en F-beta |
| Integratie met Laravel | Direct, via Composer |
| Hosting | Binnen de bestaande Laravel-app |

**Voordelen.**
- Geen extra service: één taal, één codebase, één deployment.
- Geen netwerkhop, dus geen extra vertraging en geen extra foutbron.
- De interface lijkt op die van scikit-learn, en de basisalgoritmes voor dit probleem zijn aanwezig.

**Nadelen.**
- Veel kleinere gemeenschap dan scikit-learn, met minder voorbeelden en minder antwoorden op vragen.
- Minder frequente releases; het risico bestaat dat de bibliotheek achterloopt op nieuwe PHP-versies.
- PHP is niet ontworpen voor numeriek rekenwerk. Voor snelheid bij grotere datasets is een aparte extensie
  nodig.
- ML-code en webcode lopen door elkaar, wat testen en vervangen lastiger maakt.

**Mogelijke toepassingen.** Kleine classificatie- en clusteringtaken binnen een bestaande PHP-applicatie, waar
een aparte service te veel overhead is.

### 5.3 PHP-ML

**Beschrijving.** PHP-ML is een oudere PHP-bibliotheek voor machine learning, met basisalgoritmes als
k-nearest neighbors, Naive Bayes en SVM.

**Technische specificaties.** PHP, MIT-licentie, integratie via Composer, enkele klassieke algoritmes en
eenvoudige tekstverwerking.

**Voordelen.** Eenvoudig, draait direct in Laravel.

**Nadelen.** De bibliotheek wordt al jaren niet meer actief onderhouden. Dat betekent geen fixes, geen
beveiligingsupdates en geen garantie dat hij op nieuwe PHP-versies werkt.

**Mogelijke toepassingen.** Leerdoeleinden en kleine experimenten, niet voor een applicatie die in gebruik
wordt genomen. PHP-ML valt af op knock-out-criterium K3.

### 5.4 ONNX Runtime in PHP

**Beschrijving.** ONNX is een open standaard voor het opslaan van getrainde modellen. ONNX Runtime draait zo'n
model in vrijwel elke taal. De gedachte is: train het model in Python met scikit-learn, exporteer het naar
ONNX, en voer het in PHP uit via een binding die de ONNX Runtime-bibliotheek aanroept.

**Technische specificaties.**

| Eigenschap | Waarde |
| --- | --- |
| Taal | Training in Python, uitvoering in PHP via FFI |
| Licentie | MIT (ONNX Runtime) |
| Algoritmes | Wat de exporttool ondersteunt; de meeste scikit-learn-modellen |
| Integratie met Laravel | Een PHP-binding die de native ONNX Runtime-bibliotheek laadt |
| Hosting | Modelbestand in de Laravel-app; training apart |

**Voordelen.**
- Zeer snelle uitvoering, zonder netwerkhop.
- Het beste van twee werelden: trainen met scikit-learn, uitvoeren in PHP.

**Nadelen.**
- Het model moet vooraf getraind worden. Mijn aanpak traint juist per categorie op het moment van de scan, met
  actuele data uit Open Food Facts. Met ONNX moet ik vooraf een model per categorie trainen en bijhouden.
- De voorbewerking (de tokens per product) moet in twee talen identiek zijn. Mijn eigen tokenfunctie is niet
  mee te exporteren, dus die zou ik in PHP opnieuw moeten schrijven. Twee implementaties die precies gelijk
  moeten blijven, zijn een bekende bron van fouten (Lakshmanan et al., 2020, patroon *Transform*).
- PHP heeft de FFI-extensie nodig, wat de serverconfiguratie ingewikkelder maakt.

**Mogelijke toepassingen.** Een vast, vooraf getraind model met veel aanvragen, waar elke milliseconde telt.
Dat past bij een grote productieomgeving, niet bij deze proof of concept.

### 5.5 TensorFlow/Keras en PyTorch

**Beschrijving.** De twee bekendste frameworks voor deep learning. Ze bouwen en trainen neurale netwerken, ook
voor tekst en beeld.

**Technische specificaties.** Python, licenties Apache 2.0 (TensorFlow) en BSD (PyTorch), ondersteuning voor
GPU's, en vooraf getrainde modellen voor tekst en beeld. Integratie met Laravel gaat net als bij scikit-learn
via een service.

**Voordelen.** Zeer krachtig bij grote hoeveelheden ongestructureerde data. Een vooraf getraind taalmodel kan
bijvoorbeeld ook de vrije ingrediëntentekst begrijpen in plaats van alleen de tags.

**Nadelen.** Neurale netwerken hebben veel meer data nodig dan de ~60 voorbeelden per categorie, en overfitten
anders: ze onthouden de trainingsdata in plaats van een patroon te leren. De installatie en de images zijn
groot, de training kost meer rekenkracht, en een schatting is moeilijk uit te leggen. TensorFlow en PyTorch
vallen af op knock-out-criterium K4.

**Mogelijke toepassingen.** Productherkenning op foto's (een product scannen zonder barcode), het lezen van
voedingswaarden van een etiket met OCR, of tekstclassificatie zodra er duizenden gelabelde voorbeelden zijn.

### 5.6 Beheerde clouddienst (Google Vertex AI, AWS SageMaker, Azure Machine Learning)

**Beschrijving.** De cloudaanbieder neemt training, hosting en schaling over. Met AutoML kiest de dienst zelf
een model op basis van je data. Laravel roept het model aan via een REST-API.

**Technische specificaties.** Een REST-API met authenticatie via een serviceaccount, betaling per trainingsuur
en per voorspelling, data opgeslagen bij de aanbieder, en ingebouwde monitoring van het model.

**Voordelen.** Geen eigen infrastructuur, automatische schaling, en AutoML vraagt weinig ML-kennis.

**Nadelen.** Doorlopende kosten per gebruik, afhankelijkheid van één aanbieder, data die naar de cloud gaat
(relevant voor de AVG zodra er gebruikersgegevens bij komen), en minder controle over en inzicht in het model.
Clouddiensten vallen af op knock-out-criterium K2.

**Mogelijke toepassingen.** Organisaties met budget en veel data die geen eigen ML-infrastructuur willen
beheren.

### 5.7 Taalmodel-API (OpenAI, Anthropic, Mistral)

**Beschrijving.** Een groot taalmodel krijgt een prompt zoals "Tot welke NOVA-groep behoort dit product,
gegeven de naam en ingrediënten?" en antwoordt zonder dat ik het model zelf train (zero-shot).

**Technische specificaties.** Een REST-API, betaling per verwerkt en gegenereerd token, een responstijd van
ongeveer een tot enkele seconden, en antwoorden die per aanroep kunnen verschillen.

**Voordelen.** Geen trainingsdata nodig, dus ook bruikbaar voor producten in categorieën met te weinig
gelabelde voorbeelden. Het model kan zijn antwoord in gewone taal toelichten.

**Nadelen.** Kosten per aanroep, een trage reactie voor iemand die in de winkel scant, antwoorden die niet
reproduceerbaar zijn en soms gewoon fout ("hallucinaties"), en geen betrouwbare zekerheid. De toelichting
van een taalmodel klinkt overtuigend, maar hoeft niet te zijn hoe het tot het antwoord kwam. Taalmodel-API's
vallen af op knock-out-criterium K2.

**Mogelijke toepassingen.** Een terugvaloptie voor producten waarvoor de klassieke aanpak te weinig data
heeft, of het samenvatten van productinformatie in leesbare tekst.

---

## 6. Analyse van de algoritmes binnen het gekozen component

Uit hoofdstuk 8 volgt dat scikit-learn in een Python-service het beste component is. Binnen scikit-learn
vergelijk ik vier algoritmes. Ze krijgen alle vier precies dezelfde invoer, zodat het verschil alleen in het
algoritme zit.

### 6.1 De invoer: van product naar kenmerken

Elk product wordt omgezet in een lijst kenmerken (tokens):

- de woorden uit de productnaam, zoals `name:yoghurt` en `name:aardbei`;
- de categorieën uit Open Food Facts, zoals `category:en:plain-yogurts`;
- de ingrediënten, zoals `ingredient:en:sugar` en `ingredient:en:flavouring`;
- het aantal toevoegingen (E-nummers), afgekapt op vijf, zoals `additives:3`.

TF-IDF zet die tokens om in getallen (Salton & Buckley, 1988). Een token weegt zwaarder als het in dit product
voorkomt, maar in weinig andere producten. `ingredient:en:milk` staat in bijna elke yoghurt en zegt daardoor
weinig; `ingredient:en:flavouring` is zeldzamer en wijst sterk richting bewerking. De tokens zijn ontworpen
vanuit de NOVA-definitie zelf: ultrabewerkte producten herken je aan ingrediënten en toevoegingen die je in
een gewone keuken niet gebruikt (Monteiro et al., 2019).

Het model traint per aanvraag op de producten uit de **meest specifieke categorie van het product die
minstens 20 producten met een bekende NOVA-groep heeft**. Een yoghurtdrank wordt dus vergeleken met andere
yoghurts en dranken, niet met alle producten in de database. Daardoor leert het model de verschillen
bínnen een productsoort, en dat is precies waar de NOVA-groep onderscheid maakt.

### 6.2 Majority baseline

**Werking.** Vult altijd de groep in die in de trainingsdata het vaakst voorkomt.

**Technische specificaties.** Geen model, alleen tellen. Training en voorspelling kosten niets.

**Voordelen.** Is de maatstaf: een echt model moet hier duidelijk boven uitkomen, anders voegt het niets toe.
Dit is het ML-patroon *Heuristic Benchmark* (Lakshmanan et al., 2020).

**Nadelen.** Kijkt niet naar het product zelf; zegt bij elke yoghurt hetzelfde.

**Toepassing.** Alleen als vergelijkingspunt.

### 6.3 Multinomial Naive Bayes

**Werking.** Telt per NOVA-groep hoe vaak elk kenmerk voorkomt, en berekent met de regel van Bayes welke groep
het waarschijnlijkst is gegeven de kenmerken van het product. Het model gaat ervan uit dat kenmerken
onafhankelijk van elkaar zijn; dat is de "naïeve" aanname.

**Technische specificaties.** `sklearn.naive_bayes.MultinomialNB`, met standaard Laplace-smoothing
(`alpha=1.0`), zodat een kenmerk dat in een groep nooit voorkwam niet meteen een kans van nul geeft. Trainen is
één keer tellen; gemeten trainingstijd 2 tot 3 ms.

**Voordelen.** Heel snel, werkt al met weinig voorbeelden, en is de klassieke keuze voor tekstclassificatie,
zoals spamfilters.

**Nadelen.** De onafhankelijkheidsaanname klopt hier niet: "suiker", "aroma" en "kleurstof" komen samen voor.
Het model telt zulk bewijs daardoor dubbel en geeft te zekere kansen. Bij ongelijk verdeelde groepen trekt het
sterk naar de grootste groep.

**Toepassing.** Spamfilters, sentimentanalyse, snelle eerste versies van tekstclassificatie.

### 6.4 k-Nearest Neighbours (kNN)

**Werking.** Zoekt de 5 producten in de trainingsdata die qua kenmerken het meest op dit product lijken, en
neemt hun groep over (Cover & Hart, 1967). Buren die meer lijken tellen zwaarder mee. Gelijkenis meet ik met
de cosinusgelijkenis van de TF-IDF-vectoren: hoe meer zeldzame kenmerken twee producten delen, hoe meer ze op
elkaar lijken.

**Technische specificaties.** `sklearn.neighbors.KNeighborsClassifier(n_neighbors=5, metric="cosine",
weights="distance")`. Er is geen echte training: het model onthoudt de voorbeelden (gemeten 1 ms). Bij elke
voorspelling vergelijkt het met alle voorbeelden (gemeten 0,5 ms bij ~60 voorbeelden). De rekentijd groeit dus
met de hoeveelheid data.

**Voordelen.** Intuïtief en goed uit te leggen: "dit product lijkt op deze vijf producten, en die zijn groep 4".
Het vangt patronen die niet lineair zijn, en elke nieuwe yoghurt in Open Food Facts maakt het model meteen
beter zonder opnieuw te trainen.

**Nadelen.** De zekerheid is grof: met 5 buren is het in feite een stemming, en als alle 5 dezelfde groep hebben
zegt het model 100% zeker te zijn, ook als die 5 buren zelf twijfelachtig zijn. Dubbele of foute producten in
Open Food Facts tellen direct mee. Bij grote datasets wordt het trager, omdat elke voorspelling met alle
voorbeelden vergelijkt.

**Toepassing.** Aanbevelingssystemen ("klanten die dit kochten..."), het vinden van vergelijkbare producten,
en classificatie bij kleine datasets.

### 6.5 Logistic Regression

**Werking.** Leert per NOVA-groep een gewicht voor elk kenmerk. Voor een product telt het model per groep de
gewichten van zijn kenmerken op, en zet die sommen met de softmax-functie om in een kans per groep.

**Technische specificaties.** `sklearn.linear_model.LogisticRegression(max_iter=1000)`, multinomiaal bij meer dan
twee groepen, met standaard L2-regularisatie (`C=1.0`) tegen overfitten. Gemeten trainingstijd 1,5 tot 3 ms,
voorspeltijd 0,3 ms.

**Voordelen.** Geeft kansen die redelijk betekenisvol zijn. De gewichten verklaren elke schatting: je ziet
welke kenmerken het product naar een groep duwden. Snel en stabiel.

**Nadelen.** Ziet alleen lineaire verbanden: het effect van een kenmerk is altijd even groot, ongeacht welke
andere kenmerken er zijn. Bij weinig voorbeelden van een kleine groep leert het die groep slecht.

**Toepassing.** Kredietbeoordeling, medische screening en andere toepassingen waar uitleg net zo belangrijk is
als de uitkomst.

### 6.6 Bewust niet gekozen

**Random Forest.** Sterk bij numerieke data; in het eindproduct gebruiken we het voor ontbrekende
voedingswaarden (Breiman, 2001). Met duizenden tekstkenmerken en ~60 voorbeelden overfit het snel, en uitleg per
schatting is lastiger dan bij Logistic Regression of kNN.

**Support Vector Machine.** Vaak sterk bij tekstclassificatie, maar geeft standaard geen kansen. Die zijn wel
na te rekenen, maar dat vraagt extra rekenwerk, en de uitleg is minder direct dan bij de gekozen kandidaten.

---

## 7. Design patterns

Een passend design pattern is volgens het beoordelingsformulier minimaal onderdeel van de analyse. Ik
onderscheid patronen voor de **software-architectuur** (Gamma et al., 1994) en specifieke **ML-patronen**
(Lakshmanan et al., 2020). Per patroon beschrijf ik het probleem, de oplossing, waar het in de proof of
concept zit, en welk alternatief ik heb overwogen.

```mermaid
sequenceDiagram
    participant G as Gebruiker
    participant L as Laravel (ProductController)
    participant PS as PythonService (Adapter)
    participant API as FastAPI /products/{barcode}
    participant S as ProductService
    participant OFF as OpenFoodFacts (Adapter)
    participant M as NovaModel (Strategy)
    G->>L: scant barcode
    L->>PS: findProduct(barcode)
    PS->>API: GET /products/{barcode}
    API->>S: product(barcode)
    S->>OFF: product(barcode)
    alt NOVA-groep bekend
        S-->>API: groep, estimated = false
    else NOVA-groep onbekend
        S->>OFF: category(meest specifieke categorie)
        S->>M: fit(kenmerken, groepen)
        S->>M: estimate(kenmerken van product)
        M-->>S: groep, zekerheid, redenen
    end
    API-->>PS: JSON
    PS-->>L: array
    L-->>G: productpagina
```

### 7.1 Strategy

**Probleem.** Ik wil vier algoritmes vergelijken en er één in de app gebruiken, zonder vier versies van de
service te bouwen. Ook wil ik later een beter model kunnen toevoegen zonder de rest van de code aan te passen.

**Oplossing.** Alle modellen implementeren dezelfde interface `NovaModel`, met de methodes `fit` en `estimate`,
en staan in één register (`MODELS`). De service vraagt het gekozen model op via `create_model`; welk model dat
is staat in één constante in `config.py`. De evaluatie loopt door hetzelfde register en vergelijkt alle
modellen op precies dezelfde manier.

**In de proof of concept.** `nova.py` (de interface en de vier strategieën), `service.py` (gebruikt het gekozen
model en evalueert alle modellen) en `config.py` (`CHOSEN_MODEL`).

**Overwogen alternatief.** Een `if`-structuur per modelnaam in de service. Dat werkt bij twee modellen, maar elke
nieuwe kandidaat zou de service en de evaluatie aanpassen. Met Strategy voeg je een klasse toe en verandert er
verder niets. Dat bleek direct nuttig: na de metingen in hoofdstuk 8 wisselde ik het gekozen model door één
regel in `config.py` aan te passen.

### 7.2 Adapter (Gateway)

**Probleem.** De code moet met twee externe systemen praten: Laravel met de Python-service, en de Python-service
met Open Food Facts. Die systemen kunnen veranderen of onbereikbaar zijn, en in tests wil ik ze niet echt
aanroepen.

**Oplossing.** Elk extern systeem zit achter één klasse die de rest van de code een eenvoudige, eigen
interface geeft. In Python is dat het protocol `ProductSource` met de implementatie `OpenFoodFacts`; in Laravel
is dat `PythonService`. Fouten van het externe systeem worden vertaald naar een eigen foutsoort
(`OpenFoodFactsError`, `PythonServiceException`), zodat de app er netjes op kan reageren.

**In de proof of concept.** `open_food_facts.py`, `app/Services/PythonService.php`. De Python-tests vervangen
Open Food Facts door een nep-bron; de Laravel-tests vervangen de Python-service met `Http::fake()`.

**Overwogen alternatief.** HTTP-aanroepen direct in de controller of de service. Dan zijn de tests afhankelijk
van internet en verspreidt kennis over de externe API zich door de code.

### 7.3 Microservice voor het ML-onderdeel

**Probleem.** Laravel draait PHP; de beste ML-bibliotheek draait Python.

**Oplossing.** Het ML-onderdeel is een losse service met een kleine, duidelijke API. Laravel weet niets van
modellen of Python; het vraagt alleen "geef me dit product" en krijgt JSON terug.

**In de proof of concept.** `ml-python` als FastAPI-service in een eigen container; `ml-laravel` roept hem aan
via `PYTHON_API_URL`.

**Overwogen alternatief.** ML binnen Laravel met Rubix ML (paragraaf 5.2). Minder onderdelen, maar een kleinere
bibliotheek, en ML-code en webcode zitten dan door elkaar.

### 7.4 Pipeline

**Probleem.** Tijdens training en voorspelling moeten dezelfde stappen in dezelfde volgorde worden uitgevoerd:
eerst tokens naar TF-IDF, dan het algoritme. Een fout daarin geeft verkeerde schattingen zonder foutmelding.

**Oplossing.** scikit-learn's `Pipeline` koppelt de stappen tot één object. `fit` en `predict` doorlopen altijd
dezelfde keten.

**In de proof of concept.** `SklearnModel.fit` in `nova.py`.

### 7.5 ML-patronen: Transform, Stateless Serving Function, Heuristic Benchmark, Explainable Predictions

- **Transform.** De voorbewerking die een product omzet in kenmerken, is één functie (`tokens`) die zowel bij
  training als bij voorspelling wordt gebruikt. Training en gebruik kunnen dus niet uit elkaar lopen. Dit was
  ook de doorslaggevende reden tegen ONNX Runtime (paragraaf 5.4).
- **Stateless Serving Function.** De service bewaart geen toestand tussen aanvragen: elke aanvraag haalt de
  data op, traint en schat. Daardoor kan de service zonder problemen herstarten of meerdere keren draaien.
  Alleen de antwoorden van Open Food Facts worden een uur in het geheugen bewaard, om de rate limits te sparen.
- **Heuristic Benchmark.** Elk model wordt vergeleken met de majority baseline. Zonder die vergelijking had ik
  bij frisdrank gedacht dat alle modellen uitstekend presteren (99% accuracy), terwijl de baseline net zo goed
  scoort.
- **Explainable Predictions.** Elke schatting komt met de kenmerken waarop hij is gebaseerd, en Laravel toont
  die bij het product. De gebruiker ziet dus niet alleen "groep 4 (geschat)", maar ook waarom.

---

## 8. Vergelijkende analyse

### 8.1 Componenten: knock-out-criteria

| Component | K1 Laravel | K2 Geen kosten per gebruik | K3 Onderhouden | K4 Weinig data | Door? |
| --- | --- | --- | --- | --- | --- |
| scikit-learn in Python-service | Ja (HTTP) | Ja | Ja | Ja | **Ja** |
| Rubix ML in Laravel | Ja (direct) | Ja | Ja, minder actief | Ja | **Ja** |
| PHP-ML | Ja (direct) | Ja | **Nee** | Ja | Nee |
| ONNX Runtime in PHP | Ja (FFI) | Ja | Ja | Ja | **Ja** |
| TensorFlow / PyTorch | Ja (HTTP) | Ja | Ja | **Nee** | Nee |
| Beheerde clouddienst | Ja (HTTP) | **Nee** | Ja | Ja | Nee |
| Taalmodel-API | Ja (HTTP) | **Nee** | Ja | Ja | Nee |

### 8.2 Componenten: gewogen score

Scores van 1 (slecht) tot 5 (uitstekend), met de onderbouwing uit hoofdstuk 5:

| Criterium (gewicht) | scikit-learn + FastAPI | Rubix ML | ONNX Runtime in PHP |
| --- | --- | --- | --- |
| Algoritmes en evaluatietools (20%) | 5 | 3 | 3 |
| Volwassenheid en documentatie (20%) | 5 | 2 | 4 |
| Integratie-inspanning met Laravel (20%) | 3 | 5 | 2 |
| Onderhoudbaarheid en testbaarheid (15%) | 5 | 3 | 2 |
| Aansluiting op kennis in het team (15%) | 5 | 3 | 2 |
| Prestaties (10%) | 4 | 3 | 5 |
| **Gewogen totaal (max. 5)** | **4,50** | **3,20** | **2,90** |

scikit-learn verliest op integratie, omdat er een extra service bij komt. Op alle andere criteria scoort het
gelijk of beter, en het verschil in volwassenheid, evaluatiemogelijkheden en testbaarheid is groot. Rubix ML is
een serieus alternatief als de extra service echt een probleem wordt, maar de kleinere gemeenschap is voor een
team dat ML nog leert een reëel risico.

### 8.3 Algoritmes: metingen

Ik heb de vier algoritmes gemeten op zes categorieën met Nederlandse producten uit Open Food Facts. Per
categorie heb ik 100 producten opgehaald en alleen die met een bekende NOVA-groep gebruikt. Elk algoritme is
getest met 5-voudige cross-validatie: de producten worden in vijf delen gesplitst, het model traint steeds op
vier delen en wordt getest op het vijfde, tot elk deel één keer testset was. Zo wordt elk product precies één
keer geschat door een model dat het niet eerder heeft gezien. De splitsing is vast (`random_state=42`), dus
de meting is te herhalen. De tijden zijn gemeten op mijn laptop.

**Verdeling van de groepen**

| Categorie | Met NOVA-groep | Groep 1 | Groep 3 | Groep 4 |
| --- | --- | --- | --- | --- |
| Yoghurts | 59 | 12 | 6 | 41 |
| Ontbijtgranen | 58 | 17 | 12 | 29 |
| Kaas | 58 | 0 | 25 | 33 |
| Koek | 74 | 0 | 4 | 70 |
| Brood | 63 | 3 | 16 | 44 |
| Frisdrank | 67 | 0 | 1 | 66 |

Groep 2 (bewerkte keukeningrediënten zoals olie en suiker) komt in deze categorieën niet voor. De verdeling is
in bijna elke categorie scheef, met veel groep 4. Dat bevestigt dat macro-F1 nodig is naast accuracy.

**Macro-F1 (hoger is beter; telt kleine groepen even zwaar)**

| Categorie | Baseline | Naive Bayes | kNN | Logistic Regression |
| --- | --- | --- | --- | --- |
| Yoghurts | 0,27 | 0,37 | **0,67** | 0,52 |
| Ontbijtgranen | 0,22 | 0,52 | **0,75** | 0,52 |
| Kaas | 0,36 | 0,77 | 0,80 | **0,85** |
| Koek | 0,49 | 0,49 | **0,83** | 0,49 |
| Brood | 0,27 | 0,32 | **0,51** | 0,39 |
| Frisdrank | 0,50 | 0,50 | 0,50 | 0,50 |
| **Gemiddeld** | **0,35** | **0,50** | **0,68** | **0,55** |

**Accuracy (aandeel goede schattingen)**

| Categorie | Baseline | Naive Bayes | kNN | Logistic Regression |
| --- | --- | --- | --- | --- |
| Yoghurts | 0,69 | 0,73 | **0,86** | 0,81 |
| Ontbijtgranen | 0,50 | 0,64 | **0,78** | 0,66 |
| Kaas | 0,57 | 0,79 | 0,81 | **0,86** |
| Koek | 0,95 | 0,95 | **0,97** | 0,95 |
| Brood | 0,70 | 0,71 | **0,83** | 0,75 |
| Frisdrank | 0,99 | 0,99 | 0,99 | 0,99 |
| **Gemiddeld** | **0,73** | **0,80** | **0,87** | **0,84** |

**Rekentijd (gemiddeld per categorie)**

| | Baseline | Naive Bayes | kNN | Logistic Regression |
| --- | --- | --- | --- | --- |
| Trainen | < 0,1 ms | 1,7 tot 2,7 ms | 0,8 tot 1,2 ms | 1,1 tot 3,2 ms |
| Eén schatting | < 0,01 ms | 0,3 ms | 0,5 ms | 0,3 ms |

Wat ik uit de metingen haal:

1. **Machine learning voegt iets toe.** In vijf van de zes categorieën scoort het beste model een duidelijk
   hogere macro-F1 dan de baseline, gemiddeld 0,68 tegenover 0,35.
2. **kNN is het sterkst.** Het heeft de hoogste macro-F1 in vijf van de zes categorieën en de hoogste
   gemiddelde accuracy. Logistic Regression wint alleen bij kaas.
3. **Het verschil zit in de kleine groepen.** Bij koek scoren baseline, Naive Bayes en Logistic Regression
   allemaal 0,95 accuracy, maar macro-F1 0,49: ze zeggen bij elke koek "groep 4" en missen de vier koeken in
   groep 3. kNN vindt die wel (macro-F1 0,83), omdat het per product naar de meest gelijkende voorbeelden kijkt
   in plaats van naar het gemiddelde.
4. **Bij frisdrank heeft geen enkel model meerwaarde.** Op één na is alle frisdrank groep 4, dus er valt niets
   te leren. Een model moet dan niet meer beweren dan de baseline, en dat doet het ook niet.
5. **Snelheid is geen onderscheidend criterium.** Alle modellen trainen in een paar milliseconden en schatten
   in minder dan een milliseconde. De vertraging komt van het ophalen van data bij Open Food Facts, niet van het
   model.

### 8.4 Algoritmes: gewogen score

| Criterium (gewicht) | Baseline | Naive Bayes | kNN | Logistic Regression |
| --- | --- | --- | --- | --- |
| Nauwkeurigheid (30%) | 1 | 2 | 5 | 3 |
| Uitlegbaarheid (25%) | 2 | 3 | 4 | 5 |
| Werkt met weinig data (15%) | 5 | 5 | 4 | 4 |
| Snelheid (15%) | 5 | 5 | 4 | 5 |
| Bruikbare zekerheid (10%) | 1 | 2 | 2 | 4 |
| Integratie (5%) | 5 | 5 | 5 | 5 |
| **Gewogen totaal (max. 5)** | **2,65** | **3,30** | **4,15** | **4,15** |

Onderbouwing van de scores:

- **Nauwkeurigheid:** volgt de gemiddelde macro-F1 (kNN 0,68, Logistic Regression 0,55, Naive Bayes 0,50,
  baseline 0,35).
- **Uitlegbaarheid:** Logistic Regression scoort het hoogst, omdat de gewichten precies aangeven hoeveel elk
  kenmerk bijdroeg. kNN legt uit via vergelijkbare producten en de kenmerken die het product met die buren
  deelt. Dat is begrijpelijk voor een gebruiker, maar minder precies. Naive Bayes kan per groep laten zien
  welke kenmerken vaak voorkomen, maar de dubbeltelling maakt dat minder betrouwbaar.
- **Snelheid:** kNN krijgt een 4 omdat de voorspeltijd groeit met de hoeveelheid data; voor deze PoC maakt dat
  niets uit.
- **Bruikbare zekerheid:** Logistic Regression geeft kansen die redelijk overeenkomen met hoe vaak het klopt.
  kNN geeft in feite een stemming van 5 buren, met grove stappen en soms 100%, zoals bij een haveryoghurt waarvan
  alle 5 buren groep 1 zijn.

**kNN en Logistic Regression eindigen gelijk.** Bij een gelijke stand beslist het criterium met het hoogste
gewicht, nauwkeurigheid, en daar wint kNN duidelijk: gemiddeld 0,68 tegenover 0,55 macro-F1. Dat verschil is
groot genoeg om te tellen. Bij yoghurt, ontbijtgranen en koek herkent kNN de kleinere groepen, waar Logistic
Regression ze grotendeels mist.

---

## 9. Conclusie en advies

**Component.** Ik adviseer **scikit-learn in een eigen Python-service (FastAPI)**, aangeroepen door Laravel via
HTTP. Het is het meest volwassen component, heeft alle algoritmes en evaluatietools die nodig zijn, sluit aan
op de kennis in het team, en houdt ML-code en webcode gescheiden en los testbaar. De extra service is de prijs
die we daarvoor betalen; met een Adapter aan beide kanten blijft die prijs beheersbaar.

**Algoritme.** Binnen scikit-learn kies ik **k-Nearest Neighbours** met cosinusgelijkenis op TF-IDF-kenmerken. Het
schat de NOVA-groep op echte Nederlandse productdata het best, vooral voor de kleinere groepen, en het blijft
uitlegbaar: de app laat de kenmerken zien die het product deelt met de producten waarop de schatting is
gebaseerd. Logistic Regression is een gelijkwaardig alternatief als een betrouwbaardere zekerheid belangrijker
wordt dan nauwkeurigheid. Dankzij het Strategy-patroon is het wisselen één regel in de configuratie.

**Integratie.** Laravel vraagt een product op bij de Python-service. Die haalt het product bij Open Food Facts,
traint bij een ontbrekende NOVA-groep een model op de meest specifieke categorie met minstens 20 gelabelde
producten, en stuurt de groep terug met zekerheid, redenen, het gebruikte model en het aantal
trainingsproducten. Laravel toont dat met het label "Geschat door ons model". Een bekende groep wordt nooit
overschreven, en zonder genoeg vergelijkbare producten blijft de groep onbekend.

Daarmee is de onderzoeksvraag beantwoord, en voldoet de proof of concept aan alle eisen uit hoofdstuk 3. Eén
voorbehoud: de zekerheid van kNN is grof. Paragraaf 10 beschrijft hoe dat beter kan.

---

## 10. Beperkingen en vervolg

- **De zekerheid van kNN is grof.** Een stemming van 5 buren geeft stappen van ongeveer 20%, en soms 100% waar
  dat niet terecht is. Kalibratie (in scikit-learn met `CalibratedClassifierCV`) of meer buren maakt de zekerheid
  betrouwbaarder. Tot die tijd is het eerlijker om de zekerheid te tonen als "gebaseerd op 5 vergelijkbare
  producten" dan als percentage.
- **Weinig data per categorie.** 60 producten is weinig, en de producten zónder NOVA-groep worden nu niet
  gebruikt. Met self-training kan het model ook van die producten leren (Yarowsky, 1995); in het eindproduct
  passen we dat toe.
- **Kwaliteit van Open Food Facts.** Dubbele en foutieve producten komen in de trainingsdata terecht en tellen
  bij kNN direct mee. Opschonen en ontdubbelen vóór het trainen maakt de schattingen betrouwbaarder.
- **Training per aanvraag.** Simpel en altijd actueel, maar het ophalen van een categorie bij Open Food Facts
  kost tijd en telt mee voor de rate limits. Een volgende stap is het model per categorie een tijd te bewaren.
- **Beperkte meting.** Zes categorieën is genoeg om te kiezen, maar niet om algemene uitspraken te doen. Een
  confusion matrix per categorie laat zien welke groepen het model door elkaar haalt, en meer categorieën
  maken de vergelijking sterker.
- **NOVA-labels zijn zelf niet foutloos.** De NOVA-groep in Open Food Facts wordt automatisch berekend uit
  ingrediënten die gebruikers invoeren. Het model leert dus van labels die zelf fouten kunnen bevatten.

---

## 11. Bronnen

Breiman, L. (2001). Random forests. *Machine Learning, 45*(1), 5–32. https://doi.org/10.1023/A:1010933404324

Cover, T., & Hart, P. (1967). Nearest neighbor pattern classification. *IEEE Transactions on Information Theory,
13*(1), 21–27. https://doi.org/10.1109/TIT.1967.1053964

Gamma, E., Helm, R., Johnson, R., & Vlissides, J. (1994). *Design patterns: Elements of reusable object-oriented
software*. Addison-Wesley.

Géron, A. (2022). *Hands-on machine learning with Scikit-Learn, Keras, and TensorFlow* (3e druk). O'Reilly Media.

Koren, Y., Bell, R., & Volinsky, C. (2009). Matrix factorization techniques for recommender systems. *Computer,
42*(8), 30–37. https://doi.org/10.1109/MC.2009.263

Lakshmanan, V., Robinson, S., & Munn, M. (2020). *Machine learning design patterns: Solutions to common challenges
in data preparation, model building, and MLOps*. O'Reilly Media.

Monteiro, C. A., Cannon, G., Levy, R. B., Moubarac, J.-C., Louzada, M. L. C., Rauber, F., Khandpur, N., Cediel,
G., Neri, D., Martinez-Steele, E., Baraldi, L. G., & Jaime, P. C. (2019). Ultra-processed foods: What they are and
how to identify them. *Public Health Nutrition, 22*(5), 936–941. https://doi.org/10.1017/S1368980018003762

Open Food Facts. (z.d.). *Open Food Facts API documentation*. Geraadpleegd op 8 november 2026, van
https://openfoodfacts.github.io/openfoodfacts-server/api/

Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., Blondel, M., Prettenhofer, P.,
Weiss, R., Dubourg, V., Vanderplas, J., Passos, A., Cournapeau, D., Brucher, M., Perrot, M., & Duchesnay, É.
(2011). Scikit-learn: Machine learning in Python. *Journal of Machine Learning Research, 12*, 2825–2830.

Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval. *Information Processing &
Management, 24*(5), 513–523. https://doi.org/10.1016/0306-4573(88)90021-0

Scikit-learn. (z.d.). *User guide*. Geraadpleegd op 8 november 2026, van https://scikit-learn.org/stable/user_guide.html

Yarowsky, D. (1995). Unsupervised word sense disambiguation rivaling supervised methods. In *Proceedings of the
33rd Annual Meeting of the Association for Computational Linguistics* (pp. 189–196). Association for
Computational Linguistics. https://doi.org/10.3115/981658.981684
