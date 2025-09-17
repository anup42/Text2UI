from __future__ import annotations

import random
from datetime import datetime, timedelta

from .prompt_bank import VoicePrompt


def _rng(random_seed: int | None = None) -> random.Random:
    return random.Random(random_seed)


def stub_voice_response(prompt: VoicePrompt, seed: int | None = None) -> str:
    rng = _rng(seed)
    now = datetime.now()
    category = prompt.category
    if category == "weather":
        high = rng.randint(18, 38)
        low = high - rng.randint(2, 10)
        humidity = rng.randint(30, 92)
        wind = rng.randint(4, 26)
        conditions = rng.choice([
            "Sunny",
            "Overcast",
            "Light rain",
            "Thunderstorms",
            "Humid with scattered clouds",
        ])
        response = f"{high} degC, {conditions}. High: {high} degC, Low: {low} degC. Humidity: {humidity}%, Wind: {wind} km/h."
    elif category == "calendar":
        start = now.replace(hour=rng.choice([9, 10, 11, 14, 15, 16]), minute=rng.choice([0, 15, 30, 45]), second=0, microsecond=0)
        title = rng.choice([
            "Team Sync",
            "Product Roadmap Review",
            "Client Demo",
            "Design Critique",
            "1:1 Check-in",
        ])
        next_event = start + timedelta(hours=rng.choice([1, 1.5, 2]))
        response = (
            f"Today - {now:%b %d, %Y}.\n"
            f"{start:%I:%M %p} - {title} ({rng.choice(['Zoom', 'Meet', 'Teams'])})\n"
            f"{next_event:%I:%M %p} - {rng.choice(['Deep Work Block', 'Follow-up Tasks', 'Docs Review'])}."
        )
    elif category == "music":
        track = rng.choice([
            ("Blue in Green", "Miles Davis", "Kind of Blue"),
            ("So What", "Miles Davis", "Kind of Blue"),
            ("Weightless", "Marconi Union", "Weightless"),
            ("Clair de Lune", "Claude Debussy", "Suite bergamasque"),
            ("Night Owl", "Gerry Mulligan", "Night Lights"),
        ])
        response = (
            f"Now Playing: '{track[0]}' by {track[1]}, Album: {track[2]}."
            " Options: Pause, Skip, Like, More like this."
        )
    elif category == "reminders":
        reminder_time = now.replace(hour=rng.randint(6, 22), minute=rng.choice([0, 15, 30, 45]), second=0, microsecond=0)
        response = f"Reminder set for {reminder_time:%I:%M %p}: {prompt.user_prompt.rstrip('.')}"
    elif category == "knowledge":
        subject = rng.choice([
            "Sam Altman",
            "Ada Lovelace",
            "Satya Nadella",
            "Hedy Lamarr",
            "Grace Hopper",
        ])
        response = (
            f"{subject}: {rng.choice(['Key facts ready.', 'Here is the summary.', 'Core insights collected.'])}"
            " Would you like references or latest coverage?"
        )
    elif category == "smart_home":
        device = rng.choice([
            "Living Room Lights",
            "Thermostat",
            "Garage Door",
            "Security System",
            "Sprinkler Zone 2",
        ])
        result = rng.choice([
            "adjusted",
            "turned off",
            "turned on",
            "set to eco mode",
            "locked",
        ])
        response = f"{device} {result}."
    elif category == "navigation":
        eta = rng.randint(8, 78)
        condition = rng.choice([
            "Light traffic",
            "Moderate traffic",
            "Heavy congestion",
            "Fastest route",
            "Accident reported on main highway",
        ])
        response = f"ETA {eta} minutes via primary route. {condition}."
    else:
        summary = rng.choice([
            "Captured highlights from the last sync.",
            "Draft ready - needs your approval.",
            "Summarized action items into your task board.",
            "Organized key blockers to review.",
        ])
        response = summary
    return response


def stub_ui_response(sample: dict, seed: int | None = None) -> str:
    rng = _rng(seed)
    theme = rng.choice(["light", "dark", "gradient", "card"])
    base_color = rng.choice(["#0F62FE", "#FF6F00", "#2E7D32", "#6C63FF", "#009688"])
    text_color = "#0F172A" if theme in {"light", "card"} else "#F8FAFC"
    background = "#FFFFFF" if theme == "light" else ("#0F172A" if theme == "dark" else "linear-gradient(135deg, #111827, #1E3A8A)")
    if theme == "gradient":
        card_style = "background: rgba(255, 255, 255, 0.08); border: 1px solid rgba(255, 255, 255, 0.18);"
    elif theme == "card":
        card_style = "background: #FFFFFF; box-shadow: 0 24px 68px rgba(15, 23, 42, 0.18);"
    else:
        card_style = "background: rgba(15, 23, 42, 0.85);"
    assistant_output = str(sample.get("assistant_output", "")).strip()
    assistant_html = assistant_output.replace("\n", "<br />")
    style_block = f"""
body {{
  font-family: 'Inter', sans-serif;
  background: {background};
  color: {text_color};
  margin: 0;
  padding: 32px;
}}
.card {{
  max-width: 520px;
  margin: 0 auto;
  padding: 28px;
  border-radius: 18px;
  {card_style}
}}
.title {{
  display: flex;
  align-items: center;
  gap: 12px;
  font-size: 20px;
  font-weight: 600;
  margin-bottom: 18px;
}}
.pill {{
  display: inline-flex;
  align-items: center;
  padding: 4px 10px;
  border-radius: 12px;
  font-size: 12px;
  letter-spacing: 0.02em;
  background: {base_color};
  color: #FFFFFF;
}}
.content {{
  line-height: 1.5;
  font-size: 16px;
}}
button.primary {{
  margin-top: 22px;
  padding: 12px 18px;
  border-radius: 14px;
  border: none;
  font-weight: 600;
  background: {base_color};
  color: #FFFFFF;
  cursor: pointer;
}}
""".strip()
    html = f"""
<!DOCTYPE html>
<html lang=\"en\">
  <head>
    <meta charset=\"utf-8\" />
    <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
    <title>Voice Assistant Card</title>
    <style>
{style_block}
    </style>
  </head>
  <body>
    <div class=\"card\">
      <div class=\"title\">
        <span class=\"pill\">{sample.get('category', 'assistant')}</span>
        <span>{sample.get('persona', 'User')} | {sample.get('locale', 'Unknown locale')}</span>
      </div>
      <div class=\"content\">
        <strong>User:</strong> {sample.get('user_prompt', '')}<br />
        <strong>Assistant:</strong> {assistant_html}
      </div>
      <button class=\"primary\">Close</button>
    </div>
  </body>
</html>
""".strip()
    return html
