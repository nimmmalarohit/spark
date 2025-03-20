from atlassian import Confluence

def get_page_by_space_and_title(confluence_url, username, api_token, space_name, page_title):
    """
    Get Confluence page content by space name and page title
    
    Parameters:
    - confluence_url: URL of your Confluence instance
    - username: Confluence username
    - api_token: Confluence API token
    - space_name: Name of the Confluence space
    - page_title: Title of the page to retrieve
    
    Returns:
    - Page content as HTML
    - None if page not found
    """
    # Initialize Confluence connection
    confluence = Confluence(
        url=confluence_url,
        username=username,
        password=api_token,
        cloud=True  # Set to False if using Server/Data Center version
    )
    
    # Search for the page by title in the specified space
    pages = confluence.get_all_pages_from_space(space=space_name, start=0, limit=500)
    
    # Find the page with the matching title
    for page in pages:
        if page['title'] == page_title:
            # Get the page content using the page_id
            page_content = confluence.get_page_by_id(page['id'], expand='body.storage')
            return page_content['body']['storage']['value']
    
    return None

# Example usage
if __name__ == "__main__":
    # Replace with your actual Confluence details
    confluence_url = "https://your-domain.atlassian.net"
    username = "your-email@example.com"
    api_token = "your-api-token"
    space_name = "YOURSPACENAME"
    page_title = "Your Page Title"
    
    content = get_page_by_space_and_title(
        confluence_url, 
        username, 
        api_token, 
        space_name, 
        page_title
    )
    
    if content:
        print(f"Page content retrieved successfully!")
        # Now you can process the content as needed
    else:
        print(f"Page not found in space '{space_name}' with title '{page_title}'")
