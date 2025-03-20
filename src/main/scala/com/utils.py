from atlassian import Confluence

# Initialize the Confluence connection
confluence = Confluence(
    url='https://your-domain.atlassian.net/wiki',  # Your Confluence URL
    username='your_email@example.com',             # Your email/username
    password='your_api_token'                      # Your API token (not your password)
)

# Method 1: Get content by page ID
page_id = "123456"  # Replace with your page ID
page = confluence.get_page_by_id(page_id, expand='body.storage')

# Method 2: Get content by space key and page title
space_key = "SPACE"  # Replace with your space key
page_title = "Page Title"  # Replace with your page title
page = confluence.get_page_by_title(space_key, page_title, expand='body.storage')

# Extract HTML content
if page:
    html_content = page['body']['storage']['value']
    print(html_content)
    
    # You can also get other page metadata
    print(f"Page ID: {page['id']}")
    print(f"Title: {page['title']}")
    print(f"Version: {page['version']['number']}")
else:
    print("Page not found")
