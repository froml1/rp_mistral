Global : le code doit être en anglais, les promtps aussi. Tout contenu .yaml, message doit être en lowercase et les prompt insensible à la casse. Faire un fichier d'entrée pour l'ensemble du pipeline. 

Note : chaque action de modification du corpus doit faire l'objet d'un folder. Exemple data/exports.message.json -> data/purged_message.json -> data/traducted_message.json etc. La chaine représente un workflow, si on possède déjà un fichier existant, on ne le retraite pas on le réexploite pour l'étape d'après. 

Etape 1 : Purge indicative : règles déjà établis. 

Ensuite nous traitons scène par scène :
    Chaque résultat de mistral doit être printé 
    
    ETAPE 2 : Traduction du corpus de message d'une scène vers l'anglais
    Etape 3 : Analyse primaire du contexte de la scène -> détection de thématique et subdivision du corpus et rassemblement des messages pour former des  scène supplémentaire, lorsque l'activité dans la scène semble discordante (pas le même contexte), à noté qu'on peut avoir plusieurs contexte simultanée le premier dans un, le second dans un autre. 

Etape 4 :
    Ces scènes subdivisée vont faire l'objet d'analyze spécialisée de contexte obeissant à "Qui", "Ou", "Quoi", "Quand", "Comment".

    Un fichier .yaml (plutot qu'un seul fédérant lore), va correspondre à ces informations. L'analyse doit alors intégré les informations du .yaml correspondant. 

"Qui" -> personnage : nom, prénom, surnoms, appellations, description psychologique, description physique, métier, croyances, lieux principaux.
(note : l'auteur ne peut pas être un personnage)

"Quand" -> le contexte de temps : durée, position dans une journée (indépendant du timestamp), plusieurs échelles de temps, s'il y a des gap de temps entre des fragments de la scène. 

"Ou" -> le contexte de localisation : lieu, description du lieu, est ce que le lieu change ?.

"Quoi" -> La liste claire et exhaustive du déroulé. 

"Comment" -> de quelle façon les évènements se déroule liste exhaustive, grace à quoi à qui, il s'agit des liens entre les éléments précédent. 

1 -> "Quand" done une synthèse, il ne demande aucun yaml, son résultat va servir à "Quoi".
2 -> "Ou" donne des synthèses en se reposant sur leur yaml respectif. Doit donné lieu à une synthèse spécifique du lieu et une liste d'attribut le qualifiant et une liste d'appellation à se dernier, en ignorant les pronoms ("l'oswald", "la foret de l'oswald", "le bpois de cuivre"). Il doit mettre à jour suite à l'analyse le yaml en ajoutant de l'information, corrigeant de l'incohérence.

3 -> "Qui" donne des synthèses en se reposant sur leur yaml respectif. Doit donné lieu à une synthèse spécifique du personnage et une liste d'attribut le qualifiant et une liste d'appellation à se dernier, en ignorant les pronoms ("rhys", "rhys marchal", "le masque de chrome"). Il doit mettre à jour suite à l'analyse le yaml en ajoutant de l'information, corrigeant de l'incohérence. La scène d'apparition doit être concervé dans le yaml.

3 -> "Quoi" demande en entrée non pas son yaml mais les résultats de "Qui" "Quand" "Ou", il génère des entrées dans le yaml par scène, avec un résumé de la scène et une liste exhaustive du contenu, intégrant sujet de conversation, action, etc. 

4 -> "Comment" demande l'ensemble des résultats précédent "qui" "quoi" "quand" "ou", et fait les liens entre eux par scène. Il se nourrit de son précédent yaml pour ajouté une synthèse de contexte à ces liens. 

