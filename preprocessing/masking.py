# June 7, 2024 
# Code that masks names and uncommon words in the text of the comedias
#imports

import pandas as pd
import numpy as np

import regex as re



import nltk
nltk.download('punkt')

##################################################

character_file = 'calderon-gender-prediction/all_characters.csv'

character_df = pd.read_csv(character_file, usecols = ['id','genre','character_gender','character_id', 'scenes', 'utterances', 'tokens', 'words_spoken'])

comedias_df = character_df.dropna()
comedias_df = comedias_df[comedias_df['words_spoken'] > 20]
print(comedias_df.shape)

#remove unknown gender characters
comedias_df = comedias_df[comedias_df['character_gender'] != 'UNKNOWN'] 

##################################################
#only examine comedias files, not autos, loas, or zarzuelas
comedias_df = comedias_df[(comedias_df['genre'] != 'auto sacramental') & 
                          (comedias_df['genre'] != 'loa') & 
                          (comedias_df['genre'] != 'zarzuela') & 
                          (comedias_df['genre'] != 'mojiganga')]




# ###################################################

def mask_names(comedias_df, column_name):
#define proper names - this of names & places comes from Lemann & Pado 2020:
    names = [  "Abdalá", "Abdenato", "Abén", "Abraham", "absalón", "absinto", "acaya", "acuña", 
                "adolfo", "adonías", "adonis", "alacena", "alarico", "alberto", "alcuzcuz", 
                "alecto", "alejandro", "alejo", "alfeo", "alfonso", "alfreda", "almagro", "almeida", 
                "alonso", "alpujarra", "álvarez", "álvaro", "amaltea", "aminta", "amón", "ana", "anajarte", 
                "anastasio", "andrés", "andrómeda", "anfión", "anfriso", "ángela", "ansa", "anteo", "anteros", "antioquía", "antistes",
                "antona", "antonio", "ap", "apeles", "apolo", "aquiles", "aquitofel", "aragón", "arceo", "arcombroto", "argalía", "argante",
                "argenis", "ariadna", "arias", "aristeo", "aristóbolo", "arminda", "armindo", "arnaldo", "arnesto", "arquelao", "arsidas", "ascalón", "astarot", 
                "astolfo", "astrea", "ataide", "atamas", "ataúlfo", "atenas", "atropos", "aureliano", "aurelio",
                "auristela", "aurora", "austria", "azagra", "babilonia", "ballón", "baltasar", "barcelona", "barlanzón", 
                "bartolomé", "barzoque", "basilio", "bata", "batillo", "bato","bazán", "beatriz", "becoquín", "belardo", 
                "belona", "belveder", "benavente", "benavides", "benito", "berganza", "berja", "bernardino", "bernardo", "bisiniano", 
                "blanca", "blas", "bolena", "boleno", "bolseo", "borgona", "borgoña", "bradamante", "brandemburg", "bredá", "brígida", 
                "brisac", "brito", "brunel", "brutamonte", "cadí", "calabazas", "calasiris", "calíope", "camacho", "campaspe", 
                "candaces", "candía", "candil", "cañerí", "canónigo", "capricho", "cardenio", "caribdis", "cariclea", "cariclés", 
                "carlomagno", "carlos", "cárlos", "carpoforo", "cartago", "casilda", "casimira", "casimiro", "castelví",
                "castilla", "catalina", "céfalo", "céfiro", "celandio", "celauro", "celfa", "celia", "celín", "celio", "cenobia", 
                "cenón", "centellas", "césar", "cesarino", "ceusis", "ceuta", "chacón", "chato", "chichón", "chilindrina", "chipre", 
                "cibele", "cide", "cintia", "cipriano", "ciprïano", "circe", "clara", "clariana", "clarïana", "claridiana", 
                "claridiano", "clarín", "claudio", "cleopatra", "clicie", "climene", "clodomira", "clori", "clorinda", "cloriquea", 
                "cloris", "clotaldo", "cloto", "colona", "condestable","constantinopla", "constanza",
                "copacabana", "coquín", "córdoba", "coriolano", "coruña", "cosdoras", "cosdroas", "cósdroas", "cosme", "crespo", 
                "crisanto", "cristerna", "crotaldo", "crotilda", "curcio", "dafne", "danae", "dánae", "dante", 
                "danteo", "daría", "david", "decio", "dédalo", "deidamia", "delfos", "demonio", "deyanira", "deydamia", "diana", "dïana", 
                "diego", "diógenes", "dionís", "discordia", "dominga", "dorador", "doris", "dorotea", 
                "dos", "durandarte", "eco", "efestión", "egerio", "egidio", "egipto", "egle", "egnido", "elena",
                "eleno", "eliazar", "eliud", "elvira", "emilio", "enano", "enio", "enrico", "enrique", "enriqueclori", "enríquez", 
                "ensay", "epafo", "epimeteo", "epiro", "eraclio", "ergasto", "eridano", "erídano", "eristenes", "eróstrato", "escarpin",
                "escila", "españa", "espinel", "espínola", "espolón", "estatira", "estela", "estrella", "etiopía", "eugenia", "euristio", 
                "eusebio", "fabio", "fabricio", "fadrique", "faetón", "falerina", "fauno", "febo", "federico", "fedra", "felipe", 
                "félix", "fenicio", "fénix", "fernando", "fernandov", "ferrara", "fez", "fi", "fierabrás", "figueroa", "filiberto", 
                "filipo", "filis", "fineo", "fisberto", "fitón", "fl", "flabio", "flandes", "flavio", "flérida", "flora", 
                "florante", "florencia", "florenciafederico", "floreta", "floripes","floriseo", "floro", "focas",
                "franchipán", "francia", "francisco", "friso", "galafre", "galatea", "garcés", "garcía", "gaza", "gelanor", "gil", 
                "gila", "gileta", "gilote", "ginés", "glauca", "gocia", "godmán", "golilla", "gómez", "gonzalo", "gorgias", "granada", 
                "guacolda", "guarín", "guarinos", "guáscar", "guido", "guillén", "gutierre", "guzmán", 
                "hamet", "hamete", "hebreo", "heraclio", "hércules", "hernando", "hesperia", "hianisbe", "hidalgo", "hipólita", 
                "hipólito", "hirán", "idaspes", "ifis", "ignacio", "ildefonso", "india", "inés", "inga", "íñigo", "irán", "irene", 
                "irifela", "irífile", "iris", "isaac", "isabel", "isbella", "ismenia", "israel", "italia", "jacinta", "jaques", 
                "jasón", "jebnón", "jerónimo", "jerusalén", "joab", "jonadab", "jonatás", "josef", "juan", "juana", "juanaustria", 
                "juanete", "judas", "julia", "julio", "juno", "júpiter", "justina", "justino", "laquesis", "lara", "láscaris", 
                "laura", "laurencio", "lauro", "lázaro", "lebrel", "lebrón", "lelio", "leocadia", "leogario", "león", "leonardo", 
                "leonelo", "leonido", "leonor", "lesbia", "libia", "libio", "licanor", "licanoro", "licas", "licia", "licio", "lidia", 
                "lidoro", "lindabridis", "liríope", "lirón", "lis", "lisandro", "lisarda", "lisardo", "lisboa", "lisi", "lisías", 
                "lísida", "lisidante", "lisidas", "lisipo", "livia", "locía", "lope", "lorenzo", "lotario", "loyola", "lucanor", 
                "luceyo", "lucía", "lucindo", "lucrecia", "ludovico", "luego", "luis", "luisa", "luna", "luquete", "macarandona", 
                "madama", "madrid", "magón", "mahomet", "malandrín", "malec", "maleca", "mandas", "mandinga", "manfredo", "manrique", 
                "mantua", "manuel", "marañón", "marcela", "marcelo", "marcial", "marfisa", "margarita", "margárita", "mari", "maria", 
                "maría", "mariene", "marïene", "marsilio", "marte", "matatías", "matilde", "mauricio", "máximo", "meco", "medea", 
                "medina", "medusa", "megera", "melancia", "menardes", "mencía", "méndez", "mendo", "mendoza", "menfis", "menga", 
                "menón", "mercurio", "meridián", "merlín", "milán", "milor", "mina", "minerva", "minos", "mitilene", 
                "mona", "monforte", "montpellier", "morfeo", "morgan", "morlaco", "moro", "morón", "moscatel", "moscón", "mosquito", 
                "muley", "muza", "napoles", "nápoles", "narcisa", "narciso", "nasau", "nausiclés", "neso", "ngido", 
                "ninias", "nínive", "nino", "nise", "nisida", "nísida", "normandía", "nuño",
                "ocaña", "octaviano", "octavio", "oliveros", "orbitelo", "ordoño", "oriente", "otáñez", "otavio", "otón", 
                "pablos", "palas", "pantuflo", "parma", "pasquin", "pasquín", "patacón", "patín", "patricio", "paulín", "paulo", "payo", 
                "pedro", "pelagio", "pérez", "pernía", "perote", "perseo", "persia", "persiano", "persina", "persio", "petosiris", 
                "pigmaleón", "pimentel", "pizarro", "po", "pocris", "polemón", "poliarco", "polídites", "polidoro", "polo", "polonia", 
                "ponleví", "porcia", "portugal", "posemio", "prometeo", "ramiro", "rebolledo", 
                "reinaldos", "ricardo", "ricarte", "riselo", "roberto", "roca", "rodrigo", "roldán", "roma", "romano", "rómulo", "ronda", 
                "roque", "rosa", "rosarda", "rosaura", "rosicler", "rosimunda", "rugero", "ruisellón", "ruiz", "rusia", "sabá", 
                "sabanon", "sabañón", "sabinia", "sajonia", "sabinio", "sabino", "sabinos", "salomón", "salvatierra", "sancha", 
                "sancho", "sátiro", "scipión", "sebastián", "sebastían", "segismundo", "según", "segunda", "selenisa", "seleuco", 
                "selín", "semeí", "semey", "semeyra", "semíramis", "serafina", "sergio", "sevilla", "sicilia", "sigismundo",
                "sileno", "silva", "silvia", "silvio", "simeón", "simón", "siquis", "sirena", "sirene", "siria", "siroes", 
                "siroés", "soldán", "suevia",  "tamar", "tarif", #??
                "tarudante", "teágenes", "teobaldo", "teodora", "teodoro", "teodosio", "termutes", "tesalia", "teseo", "tesífone", 
                "tetis", "tetrarca", "teuca", "teudio", "teuia", "tevia", "thomás", "tíamis", "timantes", "timoclea", "timonides", 
                "tinacria", "tiresias", "tirso", "tisbe", "toante", "toledo", "tolomeo", "tordoya", "toribio", "torrellas", "tosco", 
                "trinacria", "tristán", "tucapel", "turín", "turingia", "turpín", "tuzaní", "ulises", "urbino", "urgel", 
                "urrea", "ursino", "valencia", "velasco", "venus", "vergas", "verusa", "veturia", "vicente", 
                "violante", "volseo", "yole", "yupanguí", "zacarías", "zalamea", "zara", "zarés", "zarzuela", "zeuxis", 
                "zulemilla", "zúñiga"

                #some of the names they included in the list didn't make sense, so I've commented them out
                # "aura","talón", "tambor", "tabaco", "cupido", "flor",  "dragón", "dueña",  "fortuna", "francés",  "ó",  "bautista", 
                # "grecia", "griego", "griegos", "domingo", "castaño", "nunca", "sino", "si"
                ]
    upper_names = [name.capitalize() for name in names]


    for index, row in comedias_df.iterrows():
        #read row['tokens'] and replance any word in the list names with '[NAME]'
        for name in upper_names:
            #if the name is not directly preceded or followed by a letter, replace it with [NAME]
            row[column_name] = re.sub(r'\b' + re.escape(name) + r'\b', '[NAME]', row[column_name])
            comedias_df.at[index, column_name] = row[column_name]

    print(comedias_df[column_name])
    return comedias_df


#################################################
#mask words that appear in < 3 plays


#Add spaces between punctuation and next letter 
comedias_df['tokens'] = comedias_df['tokens'].apply(lambda x: re.sub(r'([.,?!¿¡])([A-Za-z])', r'\1 \2', x))

#any time there is an upper case in the middle of a word, add a space before it
comedias_df['tokens'] = comedias_df['tokens'].apply(lambda x: re.sub(r'([a-z])([A-Z])', r'\1 \2', x))

#remove all punctuation
comedias_df['tokens'] = comedias_df['tokens'].apply(lambda x: re.sub(r'[^\w\s]', '', x))

print(comedias_df['tokens'])

#create a dictionary where the index is the play_id and the key is every unique word from df['tokens']
play_words = {}
grouped = comedias_df.groupby('id')
for play_id, group_df in grouped:
    print(play_id)

    all_tokens = ' '.join(group_df['tokens'])
    # lower case 
    all_tokens = all_tokens.lower()
    words= nltk.word_tokenize(all_tokens)
    unique_words = set(words)
    play_words[play_id] = list(unique_words)
# print(play_words)

#count the number of plays where each word appears
word_count = {}
for play_id, words in play_words.items():
    for word in words:

        if word in word_count:
            word_count[word] += 1
        else:
            word_count[word] = 1
# print(word_count)

uncommon_words = [word for word, count in word_count.items() if count < 2]

# Compile regular expressions for each uncommon word (ignoring case)
compiled_patterns = {word: re.compile(r'\b' + re.escape(word) + r'\b', flags=re.IGNORECASE) for word in uncommon_words}

def mask_words(comedias_df, column_name):

    # Create a new column to store modified tokens
    comedias_df[f'masked_{column_name}'] = comedias_df[column_name]


    # Iterate over rows
    for index, row in comedias_df.iterrows():
        # Read row[column_name] and replace any word that's not in common_words with '[MASK]'
        for word, pattern in compiled_patterns.items():
            if word.isnumeric() == False:
            # Replace the word using the pre-compiled pattern
                row[f'masked_{column_name}'] = pattern.sub('[MASK]', row[f'masked_{column_name}'])
        comedias_df.at[index, f'masked_{column_name}'] = row[f'masked_{column_name}']
        print(row[f'masked_{column_name}'])

    # Replace the original 'tokens' column with the modified tokens
    comedias_df[column_name] = comedias_df[f'masked_{column_name}']

    # Drop the temporary 'masked_tokens' column
    comedias_df.drop(columns=[f'masked_{column_name}'], inplace=True)
    return comedias_df

print(comedias_df['tokens'])


for column_name in ['scenes','utterances','tokens']:
    print("NOW PROCESSING")
    print(column_name)

    comedias_df = mask_names(comedias_df, column_name)
    comedias_df = mask_words(comedias_df, column_name)
    

##################################################
#create a binary variable that indicates if the character is male or female

comedias_df['is_male'] = np.where(comedias_df['character_gender'] == 'MALE', 1, 0)
print(comedias_df.head())

# output the dataframe to a csv file
comedias_df.to_csv('calderon-gender-prediction/masked_comedias.csv', index=False)
