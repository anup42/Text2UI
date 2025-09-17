from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List


@dataclass
class VoicePrompt:
    category: str
    user_prompt: str
    locale: str
    persona: str


VOICE_DOMAINS: Dict[str, Dict[str, List[str]]] = {
    "weather": {
        "persona": [
            "curious commuter",
            "weekend hiker",
            "parent planning school drop-off",
            "photography enthusiast",
            "traveler with a connecting flight",
        ],
        "locales": [
            "Bangalore, India",
            "San Francisco, USA",
            "Berlin, Germany",
            "Singapore",
            "Sydney, Australia",
            "São Paulo, Brazil",
        ],
        "requests": [
            "What's the weather like this evening?",
            "Will it rain before 6 PM?",
            "Give me the UV index for today.",
            "Should I carry a jacket tomorrow morning?",
            "How's the air quality right now?",
        ],
    },
    "calendar": {
        "persona": [
            "marketing manager",
            "university student",
            "freelance designer",
            "founder juggling investor calls",
            "remote engineer collaborating across timezones",
        ],
        "locales": [
            "London, UK",
            "Austin, USA",
            "Tokyo, Japan",
            "Remote (GMT+2)",
            "Toronto, Canada",
        ],
        "requests": [
            "What meetings do I have after lunch?",
            "Block an hour for deep work tomorrow morning.",
            "Summarize my calendar for Friday.",
            "When is my next 1:1 with Alex?",
            "Do I have any conflicts next Tuesday afternoon?",
        ],
    },
    "music": {
        "persona": [
            "jazz lover",
            "lofi hip hop fan",
            "classical pianist",
            "Bollywood nostalgia seeker",
            "EDM workout playlist curator",
        ],
        "locales": [
            "Home Studio",
            "Living Room Smart Speaker",
            "CarPlay",
            "Noise-cancelling headphones",
            "Garden party setup",
        ],
        "requests": [
            "Play something mellow to help me focus.",
            "Queue up upbeat tracks for a workout.",
            "Find live recordings from the 1960s.",
            "Resume the playlist I had yesterday evening.",
            "Start a radio based on John Mayer.",
        ],
    },
    "reminders": {
        "persona": [
            "busy parent",
            "project manager",
            "medical resident",
            "graduate researcher",
            "travel coordinator",
        ],
        "locales": [
            "Kitchen Smart Display",
            "Wearable Assistant",
            "Mobile App",
            "Desktop Companion",
            "Car Dashboard",
        ],
        "requests": [
            "Remind me to send the budget update at 3 PM.",
            "Set a recurring reminder to stretch every 2 hours.",
            "Alert me when it's time to take my medication at 9 PM.",
            "Ping me tomorrow morning to water the plants.",
            "Remember to follow up with the vendor on Friday.",
        ],
    },
    "knowledge": {
        "persona": [
            "curious teen",
            "journalist on deadline",
            "history teacher",
            "startup analyst",
            "sports commentator",
        ],
        "locales": [
            "Tablet Companion",
            "Desktop Web Assistant",
            "Smart Speaker",
            "AR Glasses",
            "In-car assistant",
        ],
        "requests": [
            "Who invented the microprocessor?",
            "Summarize the latest news on renewable energy investments.",
            "How do neural networks differ from decision trees?",
            "Give me a quick biography of Ada Lovelace.",
            "What's the latest score for the Liverpool match?",
        ],
    },
    "smart_home": {
        "persona": [
            "home automation tinkerer",
            "energy-conscious homeowner",
            "vacation mode user",
            "pet owner",
            "night shift worker",
        ],
        "locales": [
            "Living room hub",
            "Bedroom smart display",
            "Mobile companion app",
            "Voice-enabled thermostat",
            "Garage console",
        ],
        "requests": [
            "Dim the hallway lights to 30 percent.",
            "Set the thermostat to 22°C until 6 AM.",
            "Turn on the backyard sprinklers for 15 minutes.",
            "Lock all downstairs doors.",
            "Switch the house to vacation mode.",
        ],
    },
    "navigation": {
        "persona": [
            "ride-share driver",
            "weekend road-tripper",
            "daily commuter",
            "cyclist",
            "delivery coordinator",
        ],
        "locales": [
            "Car dashboard",
            "Bike mount",
            "Mobile in-pocket",
            "Smartwatch",
            "Heads-up display",
        ],
        "requests": [
            "How long will it take to reach the airport with current traffic?",
            "Find the fastest route to the office avoiding tolls.",
            "Show me EV chargers along the way to Napa.",
            "Estimate arrival time if I leave in 15 minutes.",
            "What's my ETA to the downtown client site?",
        ],
    },
    "productivity": {
        "persona": [
            "note-taking writer",
            "product manager",
            "software engineer",
            "operations lead",
            "customer support agent",
        ],
        "locales": [
            "Desktop command palette",
            "Slack assistant",
            "VR workspace",
            "Wearable mic",
            "Email copilot",
        ],
        "requests": [
            "Summarize unread emails from today.",
            "Draft a thank you note to the interview panel.",
            "Highlight blockers from the sprint board.",
            "List key decisions from yesterday's design review.",
            "Capture action items from the support queue.",
        ],
    },
}


def generate_voice_queries(num_samples: int, seed: int = 42) -> List[VoicePrompt]:
    rng = random.Random(seed)
    categories = list(VOICE_DOMAINS.keys())
    prompts: List[VoicePrompt] = []
    while len(prompts) < num_samples:
        category = rng.choice(categories)
        domain = VOICE_DOMAINS[category]
        prompt = VoicePrompt(
            category=category,
            user_prompt=rng.choice(domain["requests"]),
            locale=rng.choice(domain["locales"]),
            persona=rng.choice(domain["persona"]),
        )
        prompts.append(prompt)
    return prompts
