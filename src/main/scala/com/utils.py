from atlassian import Confluence
import urllib3
from bs4 import BeautifulSoup
import re

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

def extract_confluence_content(url, username, password, space_key, page_title):
    """
    Extract and clean the content from a Confluence page
    
    Args:
        url (str): Base URL of Confluence instance
        username (str): Username for authentication
        password (str): Password for authentication
        space_key (str): Space key where the page resides
        page_title (str): Title of the page to extract
        
    Returns:
        dict: Content information including text, headings, and structured content
    """
    # Initialize Confluence connection
    confluence = Confluence(
        url=url,
        username=username,
        password=password,
        verify_ssl=False
    )
    
    try:
        # Get the page by title
        page = confluence.get_page_by_title(
            space=space_key,
            title=page_title
        )
        
        if not page:
            return {"error": f"Page '{page_title}' not found in space '{space_key}'"}
            
        page_id = page['id']
        
        # Get full page content
        content = confluence.get_page_by_id(
            page_id=page_id,
            expand='body.storage,version'
        )
        
        html_content = content['body']['storage']['value']
        
        # Parse HTML with BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Basic content extraction
        text_content = soup.get_text(separator='\n').strip()
        
        # Extract headings with their text
        headings = []
        for heading_tag in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
            for heading in soup.find_all(heading_tag):
                headings.append({
                    'level': int(heading_tag[1]),
                    'text': heading.get_text().strip()
                })
        
        # Extract tables
        tables = []
        for table in soup.find_all('table'):
            table_data = []
            # Extract headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                header_cells = header_row.find_all('th')
                headers = [cell.get_text().strip() for cell in header_cells]
            
            # Extract rows
            rows = []
            for row in table.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                if cells:
                    row_data = [cell.get_text().strip() for cell in cells]
                    rows.append(row_data)
            
            tables.append({
                'headers': headers,
                'rows': rows
            })
        
        # Extract lists
        lists = []
        for list_tag in soup.find_all(['ul', 'ol']):
            list_items = [item.get_text().strip() for item in list_tag.find_all('li')]
            lists.append({
                'type': list_tag.name,  # 'ul' for unordered, 'ol' for ordered
                'items': list_items
            })
        
        # Extract code blocks
        code_blocks = []
        for code in soup.find_all('code'):
            code_blocks.append(code.get_text())
        
        # Get page metadata
        metadata = {
            'id': page['id'],
            'title': page['title'],
            'version': content['version']['number'],
            'last_modified': content['version'].get('when', ''),
            'by': content['version'].get('by', {}).get('displayName', '')
        }
        
        return {
            'metadata': metadata,
            'text': text_content,
            'headings': headings,
            'tables': tables,
            'lists': lists,
            'code_blocks': code_blocks,
            'html': html_content  # Original HTML if needed
        }
        
    except Exception as e:
        return {"error": str(e)}

# Example usage
if __name__ == "__main__":
    confluence_url = 'https://horizon.bankofamerica.com/docs'
    username = 'your_username'
    password = 'your_password'
    space_key = 'HORIZON'
    page_title = 'JFrog Artifactory Wiki'
    
    page_content = extract_confluence_content(
        confluence_url, 
        username, 
        password, 
        space_key, 
        page_title
    )
    
    if "error" in page_content:
        print(f"Error: {page_content['error']}")
    else:
        # Print basic page info
        print(f"Page: {page_content['metadata']['title']}")
        print(f"Last modified: {page_content['metadata']['last_modified']} by {page_content['metadata']['by']}")
        print("\nHeadings:")
        for heading in page_content['headings']:
            print(f"{'  ' * (heading['level']-1)}• {heading['text']}")
        
        print("\nFirst few paragraphs:")
        paragraphs = [p for p in page_content['text'].split('\n\n') if p.strip()]
        for i, para in enumerate(paragraphs[:3]):
            if len(para) > 100:
                para = para[:100] + "..."
            print(f"{i+1}. {para}")
            
        # Print table info if available
        if page_content['tables']:
            print(f"\nContains {len(page_content['tables'])} tables")
            
        # Print list info if available
        if page_content['lists']:
            print(f"\nContains {len(page_content['lists'])} lists")
            
        # Print code block info if available
        if page_content['code_blocks']:
            print(f"\nContains {len(page_content['code_blocks'])} code blocks")
