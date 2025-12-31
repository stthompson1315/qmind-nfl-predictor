from urllib.request import urlopen
import re
import requests
url = "https://on.sportsbook.fanduel.ca/"
apiurl = "https://smp.nj.sportsbook.fanduel.com/api/sports/fixedodds/readonly/v1/getMarketPrices?priceHistory=1"

try:
    # Send a GET request to the URL
    response = requests.get(apiurl)
    
    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        print("Successfully accessed the website.")
        # Access the raw HTML content as a string
        html_content = response.text
        # You can now parse or process html_content as needed
        # print(html_content[:500]) # Print the first 500 characters for a preview
    else:
        print(f"Failed to access website. Status code: {response.status_code}")

except requests.exceptions.RequestException as e:
    print(f"An error occurred: {e}")

'''page = urlopen(url)
html_bytes = page.read()
html = html_bytes.decode("utf-8")
#html now contains all of the html for the page
print(html)'''