import survey # pip install survey
import numpy as np

def start_survey():

    # Lista di 60 domande
    questions = [
        'Stringi regolarmente nuove amicizie.',
        'Passi molto del tuo tempo libero esplorando vari argomenti casuali.',
        'Vedere altre persone piangere può facilmente farti venire voglia di piangere.',
        'Fai spesso un piano di riserva per un altro piano di riserva.',
        'Di solito rimani calmo, anche sotto forte pressione.',
        'Agli eventi sociali, raramente cerchi di presentarti a nuove persone e parli principalmente con quelle che già conosci.',
        'Preferisci completare completamente un progetto prima di iniziarne un altro.',
        'Sei molto sentimentale.',
        'Ti piace usare strumenti di organizzazione come agende e liste.',
        'Anche un piccolo errore può farti dubitare delle tue capacità e conoscenze.',
        'Ti senti a tuo agio nell’avvicinarti a qualcuno che trovi interessante e iniziare una conversazione.',
        'Non sei molto interessato a discutere varie interpretazioni e analisi di opere creative.',
        'Sei più incline a seguire la testa piuttosto che il cuore.',
        'Di solito preferisci fare quello che ti senti nel momento invece di pianificare una routine giornaliera.',
        'Ti preoccupi raramente se fai una buona impressione sulle persone che incontri.',
        'Ti piace partecipare ad attività di gruppo.',
        'Ti piacciono libri e film che ti fanno creare la tua interpretazione del finale.',
        'La tua felicità deriva più dall’aiutare gli altri a raggiungere i loro obiettivi che dai tuoi successi.',
        'Sei interessato a così tante cose che trovi difficile scegliere cosa provare per primo.',
        'Sei incline a preoccuparti che le cose possano prendere una brutta piega.',
        'Eviti i ruoli di leadership nei gruppi.',
        'Non sei affatto un tipo artistico.',
        'Pensi che il mondo sarebbe un posto migliore se le persone si affidassero più alla razionalità e meno alle emozioni.',
        'Preferisci fare le tue faccende prima di concederti un po’ di relax.',
        'Ti piace guardare le persone discutere.',
        'Tendi ad evitare di attirare l’attenzione su di te.',
        'Il tuo umore può cambiare molto rapidamente.',
        'Perdi la pazienza con le persone che non sono efficienti quanto te.',
        'Finisci spesso per fare le cose all’ultimo momento.',
        'Sei sempre stato affascinato dalla domanda su cosa, se c’è qualcosa, accade dopo la morte.',
        'Di solito preferisci stare in compagnia piuttosto che da solo.',
        'Ti annoi o perdi interesse quando la discussione diventa molto teorica.',
        'Trovi facile empatizzare con una persona le cui esperienze sono molto diverse dalle tue.',
        'Di solito rimandi la decisione finale il più a lungo possibile.',
        'Raramente metti in dubbio le scelte che hai fatto.',
        'Dopo una settimana lunga e faticosa, un evento sociale vivace è proprio quello di cui hai bisogno.',
        'Ti piace visitare i musei d’arte.',
        'Hai spesso difficoltà a capire i sentimenti degli altri.',
        'Ti piace avere una lista di cose da fare ogni giorno.',
        'Raramente ti senti insicuro.',
        'Eviti di fare telefonate.',
        'Passi spesso molto tempo a cercare di capire punti di vista molto diversi dal tuo.',
        'Nel tuo gruppo sociale, sei spesso quello che contatta gli amici e organizza attività.',
        'Se i tuoi piani vengono interrotti, la tua priorità è rimetterti in carreggiata il prima possibile.',
        'Sei ancora infastidito dagli errori che hai fatto molto tempo fa.',
        'Raramente ti soffermi a contemplare le ragioni dell’esistenza umana o il significato della vita.',
        'Le tue emozioni ti controllano più di quanto tu le controlli.',
        'Fai molta attenzione a non far fare una brutta figura agli altri, anche quando è completamente colpa loro.',
        'Il tuo stile di lavoro personale è più vicino a esplosioni spontanee di energia piuttosto che a sforzi organizzati e costanti.',
        'Quando qualcuno pensa molto bene di te, ti chiedi quanto tempo ci vorrà prima che rimanga deluso.',
        'Ti piacerebbe un lavoro che richieda di lavorare da solo per la maggior parte del tempo.',
        'Credi che riflettere su domande filosofiche astratte sia una perdita di tempo.',
        'Ti senti più attratto da luoghi con atmosfere vivaci e movimentate piuttosto che da posti tranquilli e intimi.',
        'Capisci al primo sguardo come si sente qualcuno.',
        'Ti senti spesso sopraffatto.',
        'Completi le cose in modo metodico senza saltare nessun passaggio.',
        'Sei molto incuriosito dalle cose etichettate come controverse.',
        'Lasceresti passare una buona opportunità se pensassi che qualcun altro ne ha più bisogno.',
        'Hai difficoltà a rispettare le scadenze.',
        'Ti senti sicuro che le cose andranno bene per te.'
    ]


    # Lista di opzioni di risposta con i relativi pesi
    response_options = [
        '  Molto d’accordo',                  #  3
        '  D’accordo',                        #  2
        '  Abbastanza d’accordo',             #  1
        '  Né d’accordo né in disaccordo',    #  0
        '  Abbastanza in disaccordo',         # -1
        '  Disaccordo',                       # -2
        '  Molto in disaccordo'               # -3
    ]

    # Numpy array per memorizzare le risposte
    responses = np.zeros(len(questions), dtype=int)

    # Sondaggio ...
    for i, question in enumerate(questions):
        print(f"\nDomanda {i+1}/{len(questions)}")
        index = survey.routines.select(
        survey.colors.basic("red") + question,  # Colora la domanda di giallo
        options=response_options,
        focus_mark="> ",
        evade_color=survey.colors.basic("white")  # Colora l'opzione selezionata
)
        responses[i] = 3 - index  # in base alla selezione, la risposta avrà peso compreso da 3 a -3

    print("Risposte registrate:", responses)
