import re
from langchain_core.output_parsers.base import BaseOutputParser

class PythonOutputParser(BaseOutputParser):
    """
    Extract only the contents of a ```python``` code fence, or
    return the raw text if no fence is present.
    """

    def parse(self, text: str) -> str:
        # Look for ```python ... ``` or ``` ... ```
        m = re.search(r"```(?:python)?(.*?)```", text, re.DOTALL)
        if m:
            return m.group(1).strip()
        return text.strip()
