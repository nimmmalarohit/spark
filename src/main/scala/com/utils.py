import datetime


def generate_dashboard(data):
    """
    Generate a simple HTML dashboard from MongoDB query results.
    Data should be a dictionary with query names as keys.
    This version automatically adapts to schema changes.
    """
    # Generate HTML header
    html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Analytics Dashboard</title>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .header {
                background-color: #2c3e50;
                color: white;
                padding: 15px;
                text-align: center;
                margin-bottom: 20px;
                border-radius: 5px;
            }
            .card {
                background-color: white;
                border-radius: 5px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                margin-bottom: 20px;
                overflow: hidden;
            }
            .card-header {
                background-color: #3498db;
                color: white;
                padding: 10px 15px;
                font-weight: bold;
            }
            .card-body {
                padding: 15px;
            }
            table {
                width: 100%;
                border-collapse: collapse;
            }
            th, td {
                padding: 10px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }
            th {
                background-color: #f2f2f2;
                font-weight: bold;
            }
            tr:hover {
                background-color: #f5f5f5;
            }
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Analytics Dashboard</h1>
            <div>Last Updated: """ + datetime.datetime.now().strftime("%Y-%m-%d %H:%M") + """</div>
        </div>
    """

    # For each query result, create a card with a table
    for query_name, results in data.items():
        if not results:
            continue

        # Get the first result to extract fields
        first_result = results[0] if isinstance(results, list) else results

        # Skip if not a dictionary
        if not isinstance(first_result, dict):
            continue

        # Create a card for this query
        html += f"""
        <div class="card">
            <div class="card-header">{format_title(query_name)}</div>
            <div class="card-body">
                {generate_table(results)}
            </div>
        </div>
        """

    # Close HTML
    html += """
    </body>
    </html>
    """

    return html


def generate_table(results):
    """
    Dynamically generate an HTML table from query results.
    Uses the keys from the first result to determine columns.
    """
    # Handle if results is a list or a single result
    items = results if isinstance(results, list) else [results]

    # If empty, return placeholder
    if not items:
        return "<p>No data available</p>"

    # Get the first item to extract fields
    first_item = items[0]
    if not isinstance(first_item, dict):
        return "<p>Invalid data format</p>"

    # Get field names from the first item
    fields = list(first_item.keys())

    # Generate table header
    table = "<table>\n<thead>\n<tr>"
    for field in fields:
        table += f"<th>{format_title(field)}</th>"
    table += "</tr>\n</thead>\n<tbody>"

    # Generate table rows
    for item in items:
        if not isinstance(item, dict):
            continue

        table += "\n<tr>"
        for field in fields:
            value = item.get(field, "-")
            table += f"<td>{value}</td>"
        table += "</tr>"

    table += "\n</tbody>\n</table>"
    return table


def format_title(text):
    """Format a query or field name into a readable title."""
    # Replace underscores with spaces
    formatted = text.replace('_', ' ')
    # Capitalize each word
    return ' '.join(word.capitalize() for word in formatted.split())


def create_dashboard_html(query_results):
    """
    Create the dashboard HTML from query results.
    """
    html = generate_dashboard(query_results)

    # Save to file
    with open('dashboard.html', 'w') as f:
        f.write(html)

    print("Dashboard generated and saved as 'dashboard.html'")
    return html
