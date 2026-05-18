from pathlib import Path
from pydantic import BaseModel, Field
from markitdown import MarkItDown
import asyncio


class OfficeReadArgs(BaseModel):
    path: str = Field(description="Path to the office document (docx, xlsx, pdf, pptx, etc.)")


async def office_read(args: OfficeReadArgs) -> str:
    """
    Read content from office documents (docx, xlsx, pdf, pptx, etc.) and extract as markdown.
    Supports Word documents, Excel spreadsheets, PDF files, PowerPoint presentations, and more.
    Returns the extracted text content with document title if available.
    """
    
    file_path = Path(args.path)
    
    if not file_path.exists():
        return f"Error: File not found: {args.path}"
    
    if not file_path.is_file():
        return f"Error: Not a file: {args.path}"
    
    supported_extensions = {'.docx', '.xlsx', '.pdf', '.pptx', '.doc', '.xls', '.ppt'}
    if file_path.suffix.lower() not in supported_extensions:
        return f"Error: Unsupported file format: {file_path.suffix}. Supported formats: {', '.join(sorted(supported_extensions))}"
    
    try:
        md = MarkItDown()
        result = await asyncio.to_thread(md.convert, str(file_path))
        
        title_info = f"[TITLE: {result.title}]\n\n" if hasattr(result, 'title') and result.title else ""
        content = result.text_content if hasattr(result, 'text_content') else str(result)
        
        if not content or not content.strip():
            return f"(Empty document: {args.path})"
        
        return f"{title_info}{content}"
    
    except Exception as e:
        return f"Error: Failed to read document {args.path}: {str(e)}"
