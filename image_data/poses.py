import pickle

sanskrit_english_dict = {
    "Navasana": "Boat",
    "Ardha Navasana": "Half-Boat",
    "Dhanurasana": "Bow",
    "Setu Bandha Sarvangasana": "Bridge",
    "Baddha Konasana": "Butterfly",
    "Ustrasana": "Camel",
    "Marjaryasana": "Cat",
    "Bitilasana": "Cow",
    "Utkatasana": "Chair",
    "Balasana": "Child's Pose",
    "Sivasana": "Corpse",
    "Alanasana": "Crescent Lunge",
    "Bakasana": "Crow",
    "Ardha Pincha Mayurasana": "Dolphin",
    "Adho Mukha Svanasana": "Downward-Facing Dog",
    "Garudasana": "Eagle",
    "Utthita Hasta Padangusthasana": "Extended Hand to Toe",
    "Utthita Parsvakonasana": "Extended Side Angle",
    "Pincha Mayurasana": "Forearm Stand",
    "Ardha Chandrasana": "Half-Moon",
    "Adho Mukha Vrksasana": "Handstand",
    "Anjaneyasana": "Low Lunge",
    "Supta Kapotasana": "Pigeon",
    "Eka Pada Rajakapotasana": "King Pigeon",
    "Phalakasana": "Plank",
    "Halasana": "Plow",
    "Parsvottanasana": "Pyramid",
    "Parsva Virabhadrasana": "Reverse Warrior",
    "Paschimottanasana": "Seated Forward Bend",
    "Padmasana": "Lotus",
    "Ardha Matsyendrasana": "Half Lord of the Fishes",
    "Salamba Sarvangasana": "Shoulder Stand",
    "Vasisthasana": "Side Plank",
    "Salamba Bhujangasana": "Sphinx",
    "Hanumanasana": "Splits",
    "Malasana": "Squat",
    "Uttanasana": "Standing Forward Bend", 
    "Tadasana": "Mountain",
    "Virabhadrasana One": "Warrior I",
    "Virabhadrasana Two": "Warrior II",
    "Virabhadrasana Three": "Warrior III",
    "Ashta Chandrasana": "High Lunge",
    "Camatkarasana": "Wild Thing",
    "Trikonasana": "Triangle",
    "Upavistha Konasana": "Wide-Angle Seated Forward Bend",
    "Urdhva Mukha Svsnssana": "Upward-Facing Dog",  # original spelling: Svasana
    "Urdhva Dhanurasana": "Wheel",
    "Vrksasana": "Tree",}

# save the dict
with open("sanskrit_english_dict.pkl", "wb") as f:
    pickle.dump(sanskrit_english_dict, f)

    
