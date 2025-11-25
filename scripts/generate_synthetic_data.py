import json
import random
import argparse
from pathlib import Path

FIRST_NAMES = ["aditya","neha","san", "sanjay","ramesh","pooja","ananya","tanya","arjun","yash","vikram","shreya","meera","dhruv","nikhil","pallavi","vijay","rahul","deepak","amit"]
LAST_NAMES = ["rao","patel","sharma","singh","gupta","banerjee","khan","iyer","pillai","dubey","verma","chatterjee","kulkarni","mehta","agarwal"]
CITIES = ["mumbai","delhi","bangalore","hyderabad","kolkata","jaipur","pune","chennai","surat","kochi","indore"]
LOCATIONS = ["electronic city","koramangala","banjara hills","mg road","old airport road","whitefield","powai","indiranagar","hitech city"]
DOMAINS = ["gmail","yahoo","hotmail","outlook","protonmail","rediffmail"]
TLD = ["com","co.in","in","org","co"]

NUM_WORDS = ["zero","one","two","three","four","five","six","seven","eight","nine"]

random.seed(42)


def make_person():
    fn = random.choice(FIRST_NAMES)
    ln = random.choice(LAST_NAMES)
    return f"{fn} {ln}", fn, ln


def make_phone(noisy=True):
    digits = ''.join(str(random.randint(0,9)) for _ in range(10))
    if noisy:
        if random.random() < 0.5:
            return f"{digits[:5]} {digits[5:]}"
        else:
            return ' '.join(NUM_WORDS[int(d)] for d in digits)
    else:
        return digits


def make_email(name=None):
    if name is None:
        name = random.choice(FIRST_NAMES)
    if random.random() < 0.5:
        local = name + "." + random.choice(LAST_NAMES)
    else:
        local = name
    domain = random.choice(DOMAINS)
    t = random.choice(TLD)
    if random.random() < 0.5:
        local_noisy = local.replace('.', ' dot ')
        return f"{local_noisy} at {domain} dot {t}"
    else:
        return f"{local}@{domain}.{t}"


def make_credit_card(noisy=True):
    digits = ''.join(str(random.randint(0,9)) for _ in range(16))
    if noisy and random.random() < 0.6:
        if random.random() < 0.5:
            return ' '.join([digits[i:i+4] for i in range(0,16,4)])
        else:
            return ' '.join(NUM_WORDS[int(d)] for d in digits)
    else:
        return digits


def make_date():
    day = random.randint(1,28)
    month = random.choice(["january","february","march","april","may","june","july","august","september","october","november","december"]) 
    fmt = random.random()
    year = random.randint(2023,2026)
    if fmt < 0.33:
        return f"{day}/{random.randint(1,12)}/{year}"
    elif fmt < 0.66:
        return f"{day} {month} {year}"
    else:
        return f"{day:02d}-{random.randint(1,12):02d}-{year}"


def make_example(id_idx, include_entities=True):
    parts = []
    entities = []
    cursor = 0

    intro = random.choice(["this is","my name is","i am","email id of","the office is near","i am travelling to","my phone number is"]) 
    if intro == "email id of":
        person, fn, ln = make_person()
        text = f"email id of {person} is {make_email(fn)}"
        s_person = text.index(person)
        entities.append({"start": s_person, "end": s_person + len(person), "label": "PERSON_NAME"})
        s_email = text.index('is') + 3
        email = text[s_email:]
        entities.append({"start": s_email, "end": s_email + len(email), "label": "EMAIL"})
        return {"id": f"gen_{id_idx:05d}", "text": text, "entities": entities}

    if intro == "the office is near":
        loc = random.choice(LOCATIONS)
        city = random.choice(CITIES)
        text = f"the office is near {loc} in {city} today"
        s_loc = text.index(loc)
        entities.append({"start": s_loc, "end": s_loc + len(loc), "label": "LOCATION"})
        s_city = text.index(city)
        entities.append({"start": s_city, "end": s_city + len(city), "label": "CITY"})
        return {"id": f"gen_{id_idx:05d}", "text": text, "entities": entities}

    templates = [
        "this is {person} from {city} my phone is {phone} and email is {email} we can meet on {date}",
        "my name is {person} i work in {loc} in {city}",
        "my name is {person} i am from {city} my credit card number is {cc} and it expires on {date} you can email me on {email}",
        "this is {person} my phone number is {phone} please call me tomorrow",
        "i am {person} travelling to {city} on {date}",
        "the office is near {loc} in {city} today",
    ]

    tpl = random.choice(templates)
    person, fn, ln = make_person()
    city = random.choice(CITIES)
    loc = random.choice(LOCATIONS)
    phone = make_phone()
    email = make_email(fn)
    cc = make_credit_card()
    date = make_date()

    text = tpl.format(person=person, city=city, loc=loc, phone=phone, email=email, cc=cc, date=date)

    entities = []
    if "{person}" in tpl or "person" in tpl or "{person}" in tpl:
        if "{person}" in tpl or 'person' in tpl:
            if person in text:
                s = text.index(person)
                entities.append({"start": s, "end": s + len(person), "label": "PERSON_NAME"})
    if "{city}" in tpl or city in text:
        if city in text:
            s = text.index(city)
            entities.append({"start": s, "end": s + len(city), "label": "CITY"})
    if "{loc}" in tpl and loc in text:
        s = text.index(loc)
        entities.append({"start": s, "end": s + len(loc), "label": "LOCATION"})
    if "{phone}" in tpl and phone in text:
        s = text.index(phone)
        entities.append({"start": s, "end": s + len(phone), "label": "PHONE"})
    if "{email}" in tpl and email in text:
        s = text.index(email)
        entities.append({"start": s, "end": s + len(email), "label": "EMAIL"})
    if "{cc}" in tpl and cc in text:
        s = text.index(cc)
        entities.append({"start": s, "end": s + len(cc), "label": "CREDIT_CARD"})
    if "{date}" in tpl and date in text:
        s = text.index(date)
        entities.append({"start": s, "end": s + len(date), "label": "DATE"})

    return {"id": f"gen_{id_idx:05d}", "text": text, "entities": entities}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_target", type=int, default=700)
    ap.add_argument("--dev_target", type=int, default=150)
    ap.add_argument("--data_dir", default="data")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    

    train_out = data_dir / "new_train.jsonl"
    dev_out = data_dir / "new_dev.jsonl"

    with train_out.open("w", encoding="utf-8") as tf, dev_out.open("w", encoding="utf-8") as df:
        for i in range(args.train_target):
            ex = make_example(i, include_entities=True)
            tf.write(json.dumps(ex, ensure_ascii=False) + "\n")
        for i in range(args.dev_target):
            ex = make_example(100000 + i, include_entities=True)
            df.write(json.dumps(ex, ensure_ascii=False) + "\n")

    print(f"Wrote {args.train_target} train and {args.dev_target} dev examples to {data_dir}")

if __name__ == "__main__":
    main()
