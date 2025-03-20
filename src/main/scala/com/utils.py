from atlassian import Confluence
import urllib3
import sys

# Disable SSL warnings for self-signed certificates
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Define your credentials and URL
confluence_url = 'https://horizon.tesla.com'  # Your Confluence URL
username = 'your_username'                   # Replace with your actual username
password = 'your_password'                   # Replace with your actual password

try:
    # Initialize Confluence connection
    confluence = Confluence(
        url=confluence_url,
        username=username,
        password=password,
        verify_ssl=False  # Disable SSL certificate verification
    )
    
    # Define the space and page title
    space_name = 'GCPDOC'
    page_title = 'GCP Overview'
    
    # Try a simple API call to verify connectivity
    print("Testing connection to Confluence...")
    spaces = confluence.get_all_spaces(start=0, limit=1)
    print(f"Connection successful. Found space: {spaces}")
    
    # Get page by space and title
    print(f"Attempting to find page '{page_title}' in space '{space_name}'...")
    page = confluence.get_page_by_title(
        space=space_name,
        title=page_title
    )
    
    if page:
        page_id = page['id']
        print(f"Found page with ID: {page_id}")
        
        # Get full page content
        content = confluence.get_page_by_id(
            page_id=page_id,
            expand='body.storage'
        )
        
        # Print just the first 100 characters of content to verify it works
        html_content = content['body']['storage']['value']
        preview = html_content[:100] + "..." if len(html_content) > 100 else html_content
        print(f"Page content preview: {preview}")
    else:
        print(f"Error: Page '{page_title}' not found in space '{space_name}'")
        
except Exception as e:
    print(f"Error: {e}")
    # Print more detailed error information
    import traceback
    traceback.print_exc()
    sys.exit(1)

print("Script completed successfully.")
