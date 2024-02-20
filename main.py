import logging
import sys
import typing
from datetime import date
from tempfile import NamedTemporaryFile
from urllib.request import urlretrieve
from uuid import uuid4

from tenacity import retry, wait_fixed, stop_after_attempt

import cdn
from ai import generate_prompt, generate_image
from cdn import read_public_json
from image import generate_images_for_web
from models import Days, Challenge, Word, Challenges, Day, DateEntry
from words import generate_words_for_day

DATE_FORMAT = "%Y-%m-%d"

logging.basicConfig(stream=sys.stdout, level=logging.INFO)


def get_today_str() -> str:
    return date.today().strftime(DATE_FORMAT)


# Generates a challenge for a given list of words
def create_challenge(words: list[Word], date_to_generate_for: str) -> Challenge:
    logging.info("Generating prompt")
    prompt = generate_prompt([word.word for word in words])

    logging.info("Generating image")
    generated_image_url = generate_image(prompt)

    # Download/resize/upload image
    with NamedTemporaryFile(delete=False) as image_temp_file:
        logging.info("Downloading temporary file")
        urlretrieve(generated_image_url, image_temp_file.name)

        logging.info("Processing images and generating jpg/webp files")
        images_for_web = generate_images_for_web(image_temp_file.name)

        logging.info("Uploading images to CDN")
        cdn_jpeg_url = cdn.upload_file(
            images_for_web.jpeg_path,
            f"{date_to_generate_for}/{images_for_web.jpeg_filename}",
        )
        cdn_webp_url = cdn.upload_file(
            images_for_web.webp_path,
            f"{date_to_generate_for}/{images_for_web.webp_filename}",
        )
        return Challenge(
            words=words,
            image_path=image_temp_file.name,
            image_url_jpg=cdn_jpeg_url,
            image_url_webp=cdn_webp_url,
            prompt=prompt,
        )


@retry(stop=stop_after_attempt(3), wait=wait_fixed(2 * 60))
def generate_for_date(date_to_generate_for: str):
    # Get days.json
    try:
        days_json = read_public_json(f"days.json?id={str(uuid4())}")
        days = Days.parse_obj(days_json)
    except:
        logging.error("Failed to fetch days.json, starting over with a new one")
        days = Days(days=[])

    # Get ID for today
    # TODO: Need to make it "overwrite" it if the date already exists and it was added to days.json
    challenge_id = -1
    for day in days.days:
        if day.id > challenge_id:
            challenge_id = day.id
    challenge_id += 1

    logging.info("ID assigned to date is %s", challenge_id)

    # Generate words for today
    logging.info("Generating words for today")
    words_for_day = generate_words_for_day(date_to_generate_for)
    logging.info("Words generated")

    # For each set of words, create prompt and then create/process/upload images
    # TODO: Better error handling for generating the challenges - I've gotten some 'content' errors, but since this
    # whole block is retried and sorta idempotent, should be fine?
    logging.info("Generating easy challenge")
    easy_challenge = create_challenge(words_for_day.easy, date_to_generate_for)
    medium_challenge = create_challenge(words_for_day.medium, date_to_generate_for)
    hard_challenge = create_challenge(words_for_day.hard, date_to_generate_for)
    dreaming_challenge = create_challenge(words_for_day.dreaming, date_to_generate_for)

    challenges = Challenges(
        easy=easy_challenge,
        medium=medium_challenge,
        hard=hard_challenge,
        dreaming=dreaming_challenge,
    )
    for_day = Day(date=date_to_generate_for, id=challenge_id, challenges=challenges)

    # Upload day to CDN
    logging.info("Uploading day to CDN")
    with NamedTemporaryFile(delete=False) as today_file:
        today_file.write(for_day.model_dump_json().encode("utf-8"))
        today_file.close()
        cdn.upload_file(today_file.name, f"days/{date_to_generate_for}.json")

        # Update days.json with today's data
        logging.info("Updating days file")
        days.days.append(DateEntry(id=for_day.id, date=for_day.date))
        with NamedTemporaryFile(delete=False) as new_days_file:
            new_days_file.write(days.model_dump_json().encode("utf-8"))
            new_days_file.close()
            cdn.upload_file(new_days_file.name, f"days.json")

        # If date to generate for is today, replace today.json with today's data.
        if date_to_generate_for == get_today_str():
            logging.info("Updating today's file")
            cdn.upload_file(today_file.name, "today.json")
        else:
            logging.info("Not today, not updating today.json")


def main(args: typing.Dict[str, str]):
    date_to_generate_for = args.get("date", get_today_str())
    # TODO: Validate date_to_generate_for is a date
    logging.info("Generating images for date: %s", date_to_generate_for)
    generate_for_date(date_to_generate_for)


if __name__ == "__main__":
    main({})