import requests
import pandas as pd
import re
import time
import wikipedia
from tqdm import tqdm

# ==========================================
# 1. THE DATA DICTIONARIES
# ==========================================

MASTER_ENGLISH_LITERATURE = {
    "1810s": [
        {"author": "Jane Austen", "works": ["Pride and Prejudice", "Sense and Sensibility", "Mansfield Park", "Emma", "Persuasion"]},
        {"author": "Walter Scott", "works": ["Waverley", "Ivanhoe", "Rob Roy", "The Antiquary", "Guy Mannering"]},
        {"author": "Mary Shelley", "works": ["Frankenstein", "Mathilda", "Valperga", "The Last Man", "Lodore"]},
        {"author": "Washington Irving", "works": ["The Sketch Book", "Rip Van Winkle", "The Legend of Sleepy Hollow", "Bracebridge Hall", "Tales of a Traveller"]},
        {"author": "Lord Byron", "works": ["Don Juan", "Childe Harold's Pilgrimage", "Manfred", "The Corsair", "Lara"]},
        {"author": "E.T.A. Hoffmann", "works": ["The Nutcracker", "The Sandman", "The Devil's Elixirs", "Automata", "The Golden Pot"]},
        {"author": "John Keats", "works": ["Endymion", "Hyperion", "Lamia", "Isabella", "Ode to a Nightingale"]},
        {"author": "Jacob Grimm", "works": ["Grimms' Fairy Tales Vol 1", "Grimms' Fairy Tales Vol 2", "The Frog King", "Hansel and Gretel", "Rapunzel"]},
        {"author": "Maria Edgeworth", "works": ["Castle Rackrent", "The Absentee", "Belinda", "Ormond", "Patronage"]},
        {"author": "Samuel Taylor Coleridge", "works": ["Christabel", "Kubla Khan", "Biographia Literaria", "Sibylline Leaves", "Zapolya"]}
    ],
    "1820s": [
        {"author": "James Fenimore Cooper", "works": ["The Last of the Mohicans", "The Pioneers", "The Prairie", "The Pilot", "The Red Rover"]},
        {"author": "Victor Hugo", "works": ["Hans of Iceland", "Bug-Jargal", "Odes et Ballades", "Cromwell", "The Orientals"]},
        {"author": "Thomas De Quincey", "works": ["Confessions of an English Opium-Eater", "On Murder", "The English Mail-Coach", "Suspiria de Profundis", "Joan of Arc"]},
        {"author": "Alexander Pushkin", "works": ["Boris Godunov", "Eugene Onegin", "The Prisoner of the Caucasus", "The Gypsies", "Poltava"]},
        {"author": "William Hazlitt", "works": ["Table-Talk", "The Spirit of the Age", "Liber Amoris", "Characteristics", "The Plain Speaker"]},
        {"author": "John Galt", "works": ["Annals of the Parish", "The Ayrshire Legatees", "The Provost", "The Entail", "Lawrie Todd"]},
        {"author": "Giacomo Leopardi", "works": ["Zibaldone", "Canti", "Operette Morali", "Pensieri", "Epistolario"]},
        {"author": "Lydia Maria Child", "works": ["Hobomok", "The Rebels", "The Frugal Housewife", "The Mother's Book", "Philothea"]},
        {"author": "Catharine Maria Sedgwick", "works": ["A New-England Tale", "Redwood", "Hope Leslie", "Clarence", "The Linwoods"]},
        {"author": "Honoré de Balzac", "works": ["The Chouans", "The Physiology of Marriage", "Sarrasine", "The Magic Skin", "Gobseck"]}
    ],
    "1830s": [
        {"author": "Charles Dickens", "works": ["Oliver Twist", "The Pickwick Papers", "Nicholas Nickleby", "Sketches by Boz", "Master Humphrey's Clock"]},
        {"author": "Edgar Allan Poe", "works": ["The Fall of the House of Usher", "Ligeia", "William Wilson", "The Narrative of Arthur Gordon Pym", "Berenice"]},
        {"author": "Nathaniel Hawthorne", "works": ["Twice-Told Tales", "The Minister's Black Veil", "Young Goodman Brown", "Dr. Heidegger's Experiment", "The May-Pole of Merry Mount"]},
        {"author": "Nikolai Gogol", "works": ["Taras Bulba", "The Nose", "The Overcoat", "Diary of a Madman", "Viy"]},
        {"author": "Edward Bulwer-Lytton", "works": ["The Last Days of Pompeii", "Paul Clifford", "Eugene Aram", "Rienzi", "Ernest Maltravers"]},
        {"author": "Ralph Waldo Emerson", "works": ["Nature", "The American Scholar", "Divinity School Address", "Essays: First Series", "Self-Reliance"]},
        {"author": "Frederick Marryat", "works": ["Mr. Midshipman Easy", "Peter Simple", "Jacob Faithful", "Japhet in Search of a Father", "Snarleyyow"]},
        {"author": "William Harrison Ainsworth", "works": ["Rookwood", "Jack Sheppard", "The Tower of London", "Guy Fawkes", "Old St. Paul's"]},
        {"author": "Frances Trollope", "works": ["Domestic Manners of the Americans", "The Refugee in America", "The Abbess", "Tremordyn Cliff", "The Widow Barnaby"]},
        {"author": "Letitia Elizabeth Landon", "works": ["Romance and Reality", "Francesca Carrara", "Ethel Churchill", "Duty and Inclination", "Lady Anne Granard"]}
    ],
    "1840s": [
        {"author": "Charlotte Brontë", "works": ["Jane Eyre", "Shirley", "Villette", "The Professor", "Poems by Currer, Ellis, and Acton Bell"]},
        {"author": "Emily Brontë", "works": ["Wuthering Heights", "Remembrance", "No Coward Soul Is Mine", "The Old Stoic", "A Death-Scene"]},
        {"author": "Alexandre Dumas", "works": ["The Count of Monte Cristo", "The Three Musketeers", "Twenty Years After", "The Vicomte of Bragelonne", "Queen Margot"]},
        {"author": "William Makepeace Thackeray", "works": ["Vanity Fair", "The Luck of Barry Lyndon", "The Book of Snobs", "Pendennis", "The Virginians"]},
        {"author": "Frederick Douglass", "works": ["Narrative of the Life of Frederick Douglass", "My Bondage and My Freedom", "The Heroic Slave", "Life and Times", "Abolition Fanaticism"]},
        {"author": "Edgar Allan Poe", "works": ["The Tell-Tale Heart", "The Pit and the Pendulum", "The Purloined Letter", "The Black Cat", "The Cask of Amontillado"]},
        {"author": "Nathaniel Hawthorne", "works": ["The Scarlet Letter", "The House of the Seven Gables", "The Blithedale Romance", "The Marble Faun", "Mosses from an Old Manse"]},
        {"author": "Charles Dickens", "works": ["David Copperfield", "A Christmas Carol", "Dombey and Son", "Martin Chuzzlewit", "The Chimes"]},
        {"author": "Fyodor Dostoevsky", "works": ["Poor Folk", "The Double", "White Nights", "Netochka Nezvanova", "A Weak Heart"]},
        {"author": "Anne Brontë", "works": ["The Tenant of Wildfell Hall", "Agnes Grey", "Poems", "The Narrow Way", "A Word to the 'Elect'"]}
    ],
    "1850s": [
        {"author": "Herman Melville", "works": ["Moby-Dick", "Bartleby, the Scrivener", "Benito Cereno", "Pierre", "The Confidence-Man"]},
        {"author": "Charles Dickens", "works": ["A Tale of Two Cities", "Bleak House", "Hard Times", "Little Dorrit", "Great Expectations"]},
        {"author": "Harriet Beecher Stowe", "works": ["Uncle Tom's Cabin", "Dred", "The Minister's Wooing", "The Pearl of Orr's Island", "Oldtown Folks"]},
        {"author": "Walt Whitman", "works": ["Leaves of Grass", "Song of Myself", "I Sing the Body Electric", "Out of the Cradle Endlessly Rocking", "Crossing Brooklyn Ferry"]},
        {"author": "Gustave Flaubert", "works": ["Madame Bovary", "Salammbô", "Sentimental Education", "The Temptation of Saint Anthony", "Three Tales"]},
        {"author": "Anthony Trollope", "works": ["The Warden", "Barchester Towers", "Doctor Thorne", "Framley Parsonage", "The Small House at Allington"]},
        {"author": "Elizabeth Gaskell", "works": ["Cranford", "North and South", "Ruth", "Sylvia's Lovers", "Wives and Daughters"]},
        {"author": "Ivan Turgenev", "works": ["A Sportsman's Sketches", "Rudin", "A Nest of the Gentry", "On the Eve", "Fathers and Sons"]},
        {"author": "Henry David Thoreau", "works": ["Walden", "Civil Disobedience", "Life Without Principle", "Slavery in Massachusetts", "A Plea for Captain John Brown"]},
        {"author": "Charles Baudelaire", "works": ["The Flowers of Evil", "Paris Spleen", "Artificial Paradises", "The Painter of Modern Life", "Rockets"]}
    ],
    "1860s": [
        {"author": "Fyodor Dostoevsky", "works": ["Crime and Punishment", "The Idiot", "Notes from Underground", "The Gambler", "The Possessed"]},
        {"author": "Leo Tolstoy", "works": ["War and Peace", "The Cossacks", "Childhood", "Boyhood", "Youth"]},
        {"author": "Victor Hugo", "works": ["Les Misérables", "Toilers of the Sea", "The Man Who Laughs", "Ninety-Three", "The History of a Crime"]},
        {"author": "Louisa May Alcott", "works": ["Little Women", "Little Men", "Jo's Boys", "Eight Cousins", "Rose in Bloom"]},
        {"author": "Lewis Carroll", "works": ["Alice's Adventures in Wonderland", "Through the Looking-Glass", "The Hunting of the Snark", "Jabberwocky", "Sylvie and Bruno"]},
        {"author": "George Eliot", "works": ["Silas Marner", "The Mill on the Floss", "Romola", "Felix Holt", "Middlemarch"]},
        {"author": "Jules Verne", "works": ["Journey to the Center of the Earth", "From the Earth to the Moon", "Twenty Thousand Leagues Under the Sea", "In Search of the Castaways", "The Mysterious Island"]},
        {"author": "Mary Elizabeth Braddon", "works": ["Lady Audley's Secret", "Aurora Floyd", "John Marchmont's Legacy", "The Doctor's Wife", "Henry Dunbar"]},
        {"author": "Wilkie Collins", "works": ["The Woman in White", "The Moonstone", "No Name", "Armadale", "Man and Wife"]},
        {"author": "Christina Rossetti", "works": ["Goblin Market", "The Prince's Progress", "Sing-Song", "A Pageant", "Verses"]}
    ],
    "1870s": [
        {"author": "Mark Twain", "works": ["The Adventures of Tom Sawyer", "Roughing It", "The Gilded Age", "A Tramp Abroad", "The Prince and the Pauper"]},
        {"author": "George Eliot", "works": ["Middlemarch", "Daniel Deronda", "The Impressions of Theophrastus Such", "The Legend of Jubal", "Brother Jacob"]},
        {"author": "Thomas Hardy", "works": ["Far from the Madding Crowd", "The Return of the Native", "Under the Greenwood Tree", "A Pair of Blue Eyes", "The Hand of Ethelberta"]},
        {"author": "Anna Sewell", "works": ["Black Beauty"]},
        {"author": "Henry James", "works": ["Daisy Miller", "The American", "The Europeans", "Washington Square", "The Portrait of a Lady"]},
        {"author": "Leo Tolstoy", "works": ["Anna Karenina", "A Confession", "The Death of Ivan Ilyich", "Resurrection", "The Kreutzer Sonata"]},
        {"author": "Jules Verne", "works": ["Around the World in Eighty Days", "The Mysterious Island", "Michael Strogoff", "Dick Sand", "The Begum's Fortune"]},
        {"author": "Samuel Butler", "works": ["Erewhon", "The Way of All Flesh", "Life and Habit", "Evolution, Old and New", "Unconscious Memory"]},
        {"author": "R.D. Blackmore", "works": ["Lorna Doone", "The Maid of Sker", "Alice Lorraine", "Cripps the Carrier", "Erema"]},
        {"author": "Émile Zola", "works": ["L'Assommoir", "Nana", "Germinal", "The Masterpiece", "La Bête Humaine"]}
    ],
    "1880s": [
        {"author": "Robert Louis Stevenson", "works": ["Treasure Island", "The Strange Case of Dr. Jekyll and Mr. Hyde", "Kidnapped", "The Master of Ballantrae", "The Black Arrow"]},
        {"author": "Mark Twain", "works": ["Adventures of Huckleberry Finn", "Life on the Mississippi", "A Connecticut Yankee in King Arthur's Court", "The Prince and the Pauper", "The Tragedy of Pudd'nhead Wilson"]},
        {"author": "Arthur Conan Doyle", "works": ["A Study in Scarlet", "The Sign of the Four", "Micah Clarke", "The White Company", "The Mystery of Cloomber"]},
        {"author": "Friedrich Nietzsche", "works": ["Thus Spoke Zarathustra", "Beyond Good and Evil", "On the Genealogy of Morality", "The Gay Science", "Twilight of the Idols"]},
        {"author": "Oscar Wilde", "works": ["The Happy Prince", "The Picture of Dorian Gray", "The Importance of Being Earnest", "An Ideal Husband", "Lady Windermere's Fan"]},
        {"author": "Henry James", "works": ["The Portrait of a Lady", "The Bostonians", "The Princess Casamassima", "The Aspern Papers", "The Tragic Muse"]},
        {"author": "H. Rider Haggard", "works": ["King Solomon's Mines", "She", "Allan Quatermain", "Cleopatra", "Eric Brighteyes"]},
        {"author": "Thomas Hardy", "works": ["The Mayor of Casterbridge", "The Woodlanders", "Two on a Tower", "A Laodicean", "The Trumpet-Major"]},
        {"author": "Guy de Maupassant", "works": ["Bel-Ami", "The Necklace", "Boule de Suif", "Pierre and Jean", "Une Vie"]},
        {"author": "Anton Chekhov", "works": ["The Steppe", "The Bear", "Ivanov", "A Dreary Story", "The Wood Demon"]}
    ],
    "1890s": [
        {"author": "Bram Stoker", "works": ["Dracula", "The Jewel of Seven Stars", "The Lair of the White Worm", "The Lady of the Shroud", "The Mystery of the Sea"]},
        {"author": "H.G. Wells", "works": ["The Time Machine", "The War of the Worlds", "The Invisible Man", "The Island of Doctor Moreau", "When the Sleeper Wakes"]},
        {"author": "Oscar Wilde", "works": ["The Picture of Dorian Gray", "The Importance of Being Earnest", "An Ideal Husband", "Salome", "De Profundis"]},
        {"author": "Kate Chopin", "works": ["The Awakening", "The Story of an Hour", "Desiree's Baby", "At the Cadian Ball", "The Storm"]},
        {"author": "Joseph Conrad", "works": ["Heart of Darkness", "Lord Jim", "Almayer's Folly", "The Nigger of the 'Narcissus'", "An Outcast of the Islands"]},
        {"author": "Arthur Conan Doyle", "works": ["The Hound of the Baskervilles", "The Adventures of Sherlock Holmes", "The Memoirs of Sherlock Holmes", "The Return of Sherlock Holmes", "The Lost World"]},
        {"author": "Stephen Crane", "works": ["The Red Badge of Courage", "Maggie: A Girl of the Streets", "The Open Boat", "The Blue Hotel", "The Bride Comes to Yellow Sky"]},
        {"author": "Thomas Hardy", "works": ["Tess of the d'Urbervilles", "Jude the Obscure", "The Well-Beloved", "Life's Little Ironies", "Wessex Tales"]},
        {"author": "Anton Chekhov", "works": ["Uncle Vanya", "The Seagull", "Ward No. 6", "The Lady with the Dog", "The Black Monk"]},
        {"author": "Charlotte Perkins Gilman", "works": ["The Yellow Wallpaper", "Herland", "Women and Economics", "The Home", "Moving the Mountain"]}
    ],
    "1900s": [
        {"author": "Jack London", "works": ["The Call of the Wild", "White Fang", "The Sea-Wolf", "Martin Eden", "The Iron Heel"]},
        {"author": "Upton Sinclair", "works": ["The Jungle", "Oil!", "The Brass Check", "King Coal", "Boston"]},
        {"author": "Joseph Conrad", "works": ["Nostromo", "The Secret Agent", "Under Western Eyes", "Typhoon", "Youth"]},
        {"author": "Edith Wharton", "works": ["The House of Mirth", "Ethan Frome", "The Custom of the Country", "The Age of Innocence", "Summer"]},
        {"author": "Kenneth Grahame", "works": ["The Wind in the Willows", "The Reluctant Dragon", "The Golden Age", "Dream Days", "Pagan Papers"]},
        {"author": "G.K. Chesterton", "works": ["The Man Who Was Thursday", "Orthodoxy", "The Innocence of Father Brown", "Heretics", "The Everlasting Man"]},
        {"author": "Henry James", "works": ["The Ambassadors", "The Wings of the Dove", "The Golden Bowl", "The Turn of the Screw", "The Beast in the Jungle"]},
        {"author": "E.M. Forster", "works": ["A Room with a View", "Howards End", "Where Angels Fear to Tread", "The Longest Journey", "Maurice"]},
        {"author": "L. Frank Baum", "works": ["The Wonderful Wizard of Oz", "The Marvelous Land of Oz", "Ozma of Oz", "Dorothy and the Wizard in Oz", "The Road to Oz"]},
        {"author": "Rudyard Kipling", "works": ["Kim", "Just So Stories", "Puck of Pook's Hill", "Rewards and Fairies", "Traffics and Discoveries"]}
    ],
    "1910s": [
        {"author": "James Joyce", "works": ["Dubliners", "A Portrait of the Artist as a Young Man", "Exiles", "Chamber Music", "Ulysses (Serialization)"]},
        {"author": "D.H. Lawrence", "works": ["Sons and Lovers", "The Rainbow", "Women in Love", "The White Peacock", "The Trespasser"]},
        {"author": "Franz Kafka", "works": ["The Metamorphosis", "The Trial", "The Castle", "In the Penal Colony", "A Hunger Artist"]},
        {"author": "Edgar Rice Burroughs", "works": ["Tarzan of the Apes", "A Princess of Mars", "The Gods of Mars", "The Return of Tarzan", "At the Earth's Core"]},
        {"author": "W. Somerset Maugham", "works": ["Of Human Bondage", "The Moon and Sixpence", "The Razor's Edge", "Cakes and Ale", "The Painted Veil"]},
        {"author": "Willa Cather", "works": ["O Pioneers!", "My Ántonia", "The Song of the Lark", "Alexander's Bridge", "One of Ours"]},
        {"author": "Ford Madox Ford", "works": ["The Good Soldier", "Parade's End", "The Fifth Queen", "Some Do Not...", "No More Parades"]},
        {"author": "Sherwood Anderson", "works": ["Winesburg, Ohio", "Poor White", "Dark Laughter", "The Triumph of the Egg", "Horses and Men"]},
        {"author": "John Buchan", "works": ["The Thirty-Nine Steps", "Greenmantle", "Mr Standfast", "The Three Hostages", "The Island of Sheep"]},
        {"author": "Marcel Proust", "works": ["Swann's Way", "In the Shadow of Young Girls in Flower", "The Guermantes Way", "Sodom and Gomorrah", "The Prisoner"]}
    ],
    "1920s": [
        {"author": "F. Scott Fitzgerald", "works": ["The Great Gatsby", "This Side of Paradise", "The Beautiful and Damned", "Tender Is the Night", "Tales of the Jazz Age"]},
        {"author": "Virginia Woolf", "works": ["Mrs Dalloway", "To the Lighthouse", "Orlando", "A Room of One's Own", "The Waves"]},
        {"author": "Ernest Hemingway", "works": ["The Sun Also Rises", "A Farewell to Arms", "In Our Time", "Men Without Women", "The Old Man and the Sea"]},
        {"author": "Hermann Hesse", "works": ["Steppenwolf", "Siddhartha", "Demian", "Narcissus and Goldmund", "The Glass Bead Game"]},
        {"author": "E.M. Forster", "works": ["A Passage to India", "Aspects of the Novel", "The Story of the Siren", "Pharos and Pharillon", "Abinger Harvest"]},
        {"author": "Aldous Huxley", "works": ["Point Counter Point", "Antic Hay", "Crome Yellow", "Brave New World", "Those Barren Leaves"]},
        {"author": "T.S. Eliot", "works": ["The Waste Land", "The Hollow Men", "Ash Wednesday", "Four Quartets", "Murder in the Cathedral"]},
        {"author": "Erich Maria Remarque", "works": ["All Quiet on the Western Front", "The Road Back", "Three Comrades", "Arch of Triumph", "The Black Obelisk"]},
        {"author": "William Faulkner", "works": ["The Sound and the Fury", "As I Lay Dying", "Sanctuary", "Light in August", "Absalom, Absalom!"]},
        {"author": "Theodore Dreiser", "works": ["An American Tragedy", "Sister Carrie", "The Financier", "The Titan", "The Stoic"]}
    ],
    "1930s": [
        {"author": "John Steinbeck", "works": ["The Grapes of Wrath", "Of Mice and Men", "Tortilla Flat", "In Dubious Battle", "The Red Pony"]},
        {"author": "George Orwell", "works": ["Down and Out in Paris and London", "Burmese Days", "A Clergyman's Daughter", "Keep the Aspidistra Flying", "The Road to Wigan Pier"]},
        {"author": "Aldous Huxley", "works": ["Brave New World", "Eyeless in Gaza", "After Many a Summer", "The Doors of Perception", "Island"]},
        {"author": "William Faulkner", "works": ["As I Lay Dying", "Light in August", "Absalom, Absalom!", "The Unvanquished", "The Wild Palms"]},
        {"author": "Agatha Christie", "works": ["Murder on the Orient Express", "And Then There Were None", "Death on the Nile", "The ABC Murders", "The Murder of Roger Ackroyd"]},
        {"author": "J.R.R. Tolkien", "works": ["The Hobbit", "The Fellowship of the Ring", "The Two Towers", "The Return of the King", "The Silmarillion"]},
        {"author": "Ernest Hemingway", "works": ["Death in the Afternoon", "Green Hills of Africa", "To Have and Have Not", "The Snows of Kilimanjaro", "For Whom the Bell Tolls"]},
        {"author": "Margaret Mitchell", "works": ["Gone with the Wind"]},
        {"author": "Richard Wright", "works": ["Native Son", "Uncle Tom's Children", "Black Boy", "The Outsider", "White Man, Listen!"]},
        {"author": "Graham Greene", "works": ["Brighton Rock", "The Power and the Glory", "The Heart of the Matter", "The End of the Affair", "The Quiet American"]}
    ],
    "1940s": [
        {"author": "George Orwell", "works": ["Nineteen Eighty-Four", "Animal Farm", "Homage to Catalonia", "Coming Up for Air", "Shooting an Elephant"]},
        {"author": "Albert Camus", "works": ["The Stranger", "The Plague", "The Fall", "The Myth of Sisyphus", "The Rebel"]},
        {"author": "Jean-Paul Sartre", "works": ["No Exit", "Nausea", "The Age of Reason", "The Reprieve", "Iron in the Soul"]},
        {"author": "Tennessee Williams", "works": ["A Streetcar Named Desire", "The Glass Menagerie", "Cat on a Hot Tin Roof", "Summer and Smoke", "The Rose Tattoo"]},
        {"author": "Arthur Miller", "works": ["Death of a Salesman", "The Crucible", "All My Sons", "A View from the Bridge", "The Misfits"]},
        {"author": "Jorge Luis Borges", "works": ["Ficciones", "The Aleph", "Labyrinths", "The Book of Sand", "A Universal History of Infamy"]},
        {"author": "Richard Wright", "works": ["Black Boy", "The Outsider", "Savage Holiday", "The Long Dream", "Eight Men"]},
        {"author": "Carson McCullers", "works": ["The Heart Is a Lonely Hunter", "Reflections in a Golden Eye", "The Member of the Wedding", "The Ballad of the Sad Cafe", "Clock Without Hands"]},
        {"author": "Norman Mailer", "works": ["The Naked and the Dead", "Barbary Shore", "The Deer Park", "An American Dream", "The Armies of the Night"]},
        {"author": "James A. Michener", "works": ["Tales of the South Pacific", "Hawaii", "Centennial", "The Source", "Space"]}
    ],
    "1950s": [
        {"author": "J.D. Salinger", "works": ["The Catcher in the Rye", "Nine Stories", "Franny and Zooey", "Raise High the Roof Beam, Carpenters", "Seymour: An Introduction"]},
        {"author": "Ray Bradbury", "works": ["Fahrenheit 451", "The Martian Chronicles", "Something Wicked This Way Comes", "Dandelion Wine", "The Illustrated Man"]},
        {"author": "William Golding", "works": ["Lord of the Flies", "The Inheritors", "Pincher Martin", "Free Fall", "The Spire"]},
        {"author": "Vladimir Nabokov", "works": ["Lolita", "Pale Fire", "Pnin", "Speak, Memory", "Ada, or Ardor"]},
        {"author": "Jack Kerouac", "works": ["On the Road", "The Dharma Bums", "Desolation Angels", "Big Sur", "Visions of Cody"]},
        {"author": "James Baldwin", "works": ["Go Tell It on the Mountain", "Giovanni's Room", "Another Country", "The Fire Next Time", "Notes of a Native Son"]},
        {"author": "Ralph Ellison", "works": ["Invisible Man", "Shadow and Act", "Going to the Territory", "Juneteenth", "Flying Home"]},
        {"author": "Samuel Beckett", "works": ["Waiting for Godot", "Endgame", "Krapp's Last Tape", "Happy Days", "Molloy"]},
        {"author": "Ernest Hemingway", "works": ["The Old Man and the Sea", "A Moveable Feast", "Islands in the Stream", "The Garden of Eden", "True at First Light"]},
        {"author": "Arthur C. Clarke", "works": ["Childhood's End", "The City and the Stars", "Rendezvous with Rama", "2001: A Space Odyssey", "The Fountains of Paradise"]}
    ],
    "1960s": [
        {"author": "Harper Lee", "works": ["To Kill a Mockingbird", "Go Set a Watchman"]},
        {"author": "Joseph Heller", "works": ["Catch-22", "Something Happened", "Good as Gold", "God Knows", "Closing Time"]},
        {"author": "Kurt Vonnegut", "works": ["Slaughterhouse-Five", "Cat's Cradle", "Breakfast of Champions", "Mother Night", "The Sirens of Titan"]},
        {"author": "Frank Herbert", "works": ["Dune", "Dune Messiah", "Children of Dune", "God Emperor of Dune", "Heretics of Dune"]},
        {"author": "Thomas Pynchon", "works": ["The Crying of Lot 49", "V.", "Gravity's Rainbow", "Vineland", "Mason & Dixon"]},
        {"author": "Anthony Burgess", "works": ["A Clockwork Orange", "Earthly Powers", "The Wanting Seed", "Nothing Like the Sun", "Inside Mr. Enderby"]},
        {"author": "Sylvia Plath", "works": ["The Bell Jar", "Ariel", "The Colossus", "Crossing the Water", "Winter Trees"]},
        {"author": "Ken Kesey", "works": ["One Flew Over the Cuckoo's Nest", "Sometimes a Great Notion", "Demon Box", "Sailor Song", "Last Go Round"]},
        {"author": "Ursula K. Le Guin", "works": ["The Left Hand of Darkness", "A Wizard of Earthsea", "The Dispossessed", "The Lathe of Heaven", "The Tombs of Atuan"]},
        {"author": "Philip K. Dick", "works": ["Do Androids Dream of Electric Sheep?", "The Man in the High Castle", "Ubik", "A Scanner Darkly", "The Three Stigmata of Palmer Eldritch"]}
    ]
}

# Expanded Wiki topics to ensure massive text yields


# ==========================================
# 2. GUTENBERG EXTRACTION LOGIC
# ==========================================
def get_gutenberg_id(author, title):
    search_query = f"{author} {title}".replace(" ", "%20")
    api_url = f"https://gutendex.com/books?search={search_query}"
    try:
        response = requests.get(api_url, timeout=10).json()
        if response['count'] > 0: 
            return response['results'][0]['id']
    except: 
        pass
    return None

# ==========================================
# 3. THE EXTRACTION ENGINE
# ==========================================
def extract_master_literature():
    results = []
    MAX_PARAGRAPHS_PER_BOOK = 60 # High quota for massive data yield
    
    # Browser stealth header to bypass Project Gutenberg's anti-bot system
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
    }
    
    print("\n" + "="*60)
    print("🚀 INITIALIZING PROJECT GUTENBERG LITERATURE SCRAPER")
    print("="*60)
    
    # NEW LOOP LOGIC to handle the "works" array
    for decade, authors_list in MASTER_ENGLISH_LITERATURE.items():
        print(f"\n📚 Processing {decade}...")
        
        for author_data in authors_list:
            author_name = author_data['author']
            
            for title in author_data['works']:
                
                # 1. Fetch the Book ID
                book_id = get_gutenberg_id(author_name, title)
                
                if not book_id:
                    print(f"  ❌ Not in Gutenberg (Likely Copyright): {title} by {author_name}")
                    continue
                    
                # 2. Download the Text
                url = f"https://www.gutenberg.org/ebooks/{book_id}.txt.utf-8"
                try:
                    response = requests.get(url, headers=headers, timeout=15)
                    if response.status_code != 200: 
                        continue
                        
                    text = response.text.replace('\r', '') 
                    
                    # Clean boilerplate
                    if "*** START OF" in text and "*** END OF" in text:
                        text = text.split("*** START OF")[1].split("*** END OF")[0]
                        text = text.split("***", 1)[-1] 
                    
                    # 3. Extract Paragraphs
                    count = 0
                    for p in text.split('\n\n'):
                        clean_p = re.sub(r'\s+', ' ', p).strip()
                        
                        # Substantive paragraph filter
                        if 150 < len(clean_p) < 1000:
                            results.append({
                                "author": author_name,
                                "title": title,
                                "time_period": decade, 
                                "language": "en", 
                                "source": "gutenberg_fiction", 
                                "text": clean_p
                            })
                            count += 1
                            if count >= MAX_PARAGRAPHS_PER_BOOK: 
                                break
                                
                    print(f"  ✅ Extracted {count} rows: {title} ({book_id})")
                    time.sleep(1.5) # Crucial to prevent IP ban
                    
                except Exception as e: 
                    print(f"  ⚠️ Network Error on {title}: {e}")

    # ==========================================
    # 4. SAVE AND EXPORT
    # ==========================================
    df = pd.DataFrame(results)
    
    # Shuffle the data to mix authors and decades randomly (good for ML training)
    if not df.empty:
        df = df.sample(frac=1, random_state=42).reset_index(drop=True)
        
        output_file = "master_english_fiction_corpus.csv"
        df.to_csv(output_file, index=False, encoding='utf-8-sig')
        
        print("\n" + "="*60)
        print(f"🎉 SUCCESS! Saved to {output_file}")
        print(f"Total High-Quality Fiction Rows: {len(df)}")
        print("="*60)
    else:
        print("\n⚠️ No data was extracted. Check network connection.")

if __name__ == "__main__":
    extract_master_literature()