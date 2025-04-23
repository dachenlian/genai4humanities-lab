from pathlib import Path
from collections.abc import Sequence

from PIL import Image
from pydantic import BaseModel, Field


class QuestionAnswer(BaseModel):
    question: str = Field(..., description="The question to be answered.")
    answer: str = Field(..., description="The answer to the question.")


class Page(BaseModel):
    text: str = Field(..., description="The text content of the page.")
    img: str = Field(..., description="The image file name of the page.")


class Story(BaseModel):
    title: str = Field(default="Untitled", description="The title of the story.")
    pages: dict[str, Page] = Field(
        ...,
        description="A dictionary of pages, where the key is the page number and the value is a Page object.",
    )

    def gather_text_and_images(self, img_path: str | Path) -> list[str | Image.Image]:
        out = []
        img_path = Path(img_path)
        for p_name, page in self.pages.items():
            out.append(f"Page {p_name}")
            if page.img:
                img = img_path / page.img
                if img.exists():
                    out.append(Image.open(img))
            if page.text:
                out.append(page.text)
        return out
