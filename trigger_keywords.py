# trigger_keywords.py

COMMON_TRIGGERS = {
    "cold": ["cold", "chilly", "freezing", "low temperature"],
    "stress": ["stress", "overwhelmed", "pressure", "anxious", "panic", "tense"],
    "dehydration": ["thirsty", "no water", "dehydrated", "dry"],
    "exertion": ["exercise", "worked out", "overdid", "fatigued", "strain"],
    "missed_meds": ["missed meds", "forgot meds", "no medicine", "skipped meds"],
    "anger": ["angry", "mad", "furious", "frustrated"],
    "lack_of_sleep": ["tired", "no sleep", "restless", "insomnia", "couldn't sleep"],
    "hunger": ["hungry", "no food", "skipped meal"]
}

def detect_triggers(text):
    """
    Detect potential triggers in user-provided free text based on predefined keyword lists.

    Args:
        text (str): User input text (e.g., symptom log or journal entry)

    Returns:
        list: List of detected trigger labels (e.g., ['stress', 'cold'])
    """
    text = text.lower()
    detected = []
    for trigger, keywords in COMMON_TRIGGERS.items():
        if any(keyword in text for keyword in keywords):
            detected.append(trigger)
    return detected
