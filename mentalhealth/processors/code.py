from typing import Dict
class CodeProcessor:  
    @staticmethod
    def process_code(code: str, language: str = "python") -> Dict:
        lines = code.split('\n')
        non_empty_lines = [l for l in lines if l.strip()]
        
        comment_chars = {'python': '#', 'javascript': '//', 'java': '//', 'c++': '//'}
        comment_char = comment_chars.get(language, '#')
        comments = [l for l in non_empty_lines if l.strip().startswith(comment_char)]
        
        comment_text = " ".join([
            l.strip().lstrip(comment_char).strip() 
            for l in comments
        ])
        
        functions = len([l for l in non_empty_lines if 'def ' in l or 'function ' in l])
        classes = len([l for l in non_empty_lines if 'class ' in l])
        
        return {
            "language": language,
            "total_lines": len(lines),
            "code_lines": len(non_empty_lines),
            "comment_lines": len(comments),
            "comment_text": comment_text,
            "functions": functions,
            "classes": classes,
            "complexity": "simple" if len(non_empty_lines) < 20 else "moderate" if len(non_empty_lines) < 50 else "complex"
        }
    
    @staticmethod
    def generate_description(code: str, language: str = "python", metadata: Dict = None) -> str:
        if metadata is None:
            metadata = CodeProcessor.process_code(code, language)
        
        desc = f"{metadata['language']} code with {metadata['complexity']} complexity. "
        desc += f"{metadata['code_lines']} lines of code, {metadata['functions']} functions. "
        
        if metadata.get('comment_text'):
            desc += f"Comments: {metadata['comment_text'][:200]}"
        
        return desc