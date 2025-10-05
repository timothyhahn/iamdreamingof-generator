from uuid import uuid4

from pydantic import BaseModel
from wand.image import Image


class ImagesForWeb(BaseModel):
    jpeg_path: str
    jpeg_filename: str
    webp_path: str
    webp_filename: str


def generate_images_for_web(filename: str) -> ImagesForWeb:
    output_uuid = str(uuid4())
    jpeg_filename = f"{output_uuid}.jpg"
    webp_filename = f"{output_uuid}.webp"
    jpeg_path = f"/tmp/{jpeg_filename}"
    webp_path = f"/tmp/{webp_filename}"

    with Image(filename=filename) as img:
        # Generate JPEG
        with img.clone() as i:
            i.resize(800, 800)
            i.format = "jpg"
            i.save(filename=jpeg_path)

        # Generate WebP
        with img.clone() as i:
            i.resize(800, 800)
            i.format = "webp"
            i.save(filename=webp_path)

    return ImagesForWeb(
        jpeg_path=jpeg_path,
        webp_path=webp_path,
        jpeg_filename=jpeg_filename,
        webp_filename=webp_filename,
    )
