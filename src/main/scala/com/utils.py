from atlassian import Confluence
import urllib3

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your credentials and correct URL
confluence_url = 'https://horizon.bankofamerica.com'  # Base URL without /docs
username = 'your_username'
password = 'your_password'

try:
    # Initialize Confluence connection
    confluence = Confluence(
        url=confluence_url,
        username=username,
        password=password,
        verify_ssl=False
    )
    
    # Test connection with proper space key
    print("Testing connection to Confluence...")
    
    # From your URL, the space key appears to be "HORIZON"
    space_key = 'HORIZON'
    
    # Get space info to verify connection
    space_info = confluence.get_space(space_key)
    print(f"Successfully connected to space: {space_info['name']}")
    
    # To get the specific page you're trying to access
    page_title = 'JFrog Artifactory Wiki'  # Based on your URL
    
    page = confluence.get_page_by_title(
        space=space_key,
        title=page_title
    )
    
    if page:
        page_id = page['id']
        print(f"Found page with ID: {page_id}")
        
        content = confluence.get_page_by_id(
            page_id=page_id,
            expand='body.storage'
        )
        
        # Print just the first 100 characters of content
        html_content = content['body']['storage']['value']
        preview = html_content[:100] + "..." if len(html_content) > 100 else html_content
        print(f"Page content preview: {preview}")
    else:
        print(f"Page '{page_title}' not found in space '{space_key}'")
        
except Exception as e:
    print(f"Error: {str(e)}")
    import traceback
    traceback.print_exc()

print("Script completed.")
