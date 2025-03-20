from atlassian import Confluence
import urllib3

# Disable SSL warnings (since you're using self-signed certificates)
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize Confluence connection with hardcoded credentials
confluence = Confluence(
    url='https://horizon.tesla.com',  # Your Confluence URL
    username='your_username',         # Your username
    password='your_password',         # Your password
    verify_ssl=False                  # Disable SSL verification
)

# Get page by space and title (for the GCP Overview page)
space_name = 'GCPDOC'
page_title = 'GCP Overview'

try:
    # Get the page by space and title
    page = confluence.get_page_by_title(
        space=space_name,
        title=page_title
    )
    
    if page:
        # Get full page content with body
        page_id = page['id']
        content = confluence.get_page_by_id(
            page_id=page_id,
            expand='body.storage'
        )
        
        # Print the page content (HTML)
        print(content['body']['storage']['value'])
    else:
        print(f"Page '{page_title}' not found in space '{space_name}'")
        
except Exception as e:
    print(f"Error accessing Confluence: {e}")
