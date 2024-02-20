from uuid import uuid4

from pydantic import BaseModel
from wand.image import Image


class ImagesForWeb(BaseModel):
    jpeg_path: str
    jpeg_filename: str
    webp_path: str
    webp_filename: str


def generate_images_for_web(filename: str) -> ImagesForWeb:
    jpeg_path = None
    webp_path = None
    jpeg_filename = None
    webp_filename = None
    output_uuid = str(uuid4())

    with Image(filename=filename) as img:
        for file_format in ["jpg", "webp"]:
            with img.clone() as i:
                output_name = f"{output_uuid}.{file_format}"
                output_path = f"/tmp/output_name"
                i.resize(800, 800)
                i.format = file_format
                i.save(filename=output_path)
                if file_format == "jpg":
                    jpeg_path = output_path
                    jpeg_filename = output_name
                else:
                    webp_path = output_path
                    webp_filename = output_name

    return ImagesForWeb(
        jpeg_path=jpeg_path,
        webp_path=webp_path,
        jpeg_filename=jpeg_filename,
        webp_filename=webp_filename,
    )
