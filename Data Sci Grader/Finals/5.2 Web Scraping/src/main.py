from bs4 import BeautifulSoup


# Function to get open files
def get_soup(file_path):
    with open(file_path, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    return soup


# Example div:
# <tr class="country">
#     <td>Sunday</td>
#     <td style="white-space: nowrap">
#         <time itemprop="startDate" datetime="2023-01-01">Jan 01</time>
#     </td>
#     <td>
#         <a
#             class="country-listing"
#             title="Thailand"
#             href="https://www.officeholidays.com/holidays/thailand/international-new-years-day"
#             >New Year&#039;s Day</a
#         >
#     </td>
#     <td style="white-space: nowrap" class="comments">
#         National Holiday
#     </td>
#     <td class="hide-ipadmobile"></td>
# </tr>


def Q1(file_path):
    soup = get_soup(file_path)
    # Create an empty dict to store the frequencies of each day
    day_freq = {
        "Monday": 0,
        "Tuesday": 0,
        "Wednesday": 0,
        "Thursday": 0,
        "Friday": 0,
        "Saturday": 0,
        "Sunday": 0,
    }

    parent = soup.find_all("tr", class_="country")
    for p in parent:
        for day in day_freq.keys():
            if day in p.find("td"):
                day_freq[day] += 1

    return list(day_freq.values())


def Q2(file_path):
    soup = get_soup(file_path)
    parent = soup.find_all("tr", class_="country")
    for p in parent:
        if "Visakha Bucha Day" in p.get_text().strip():
            return p.find("td").get_text().strip()


# Submission for Grader
# exec(input().strip())


# Testing
def check():
    file_path = "Data Sci Grader/Finals/5.2 Web Scraping/src/data/website.html"
    # https://dsde.nattee.net/problems/77/get_attachment
    print(Q1(file_path))
    print(Q2(file_path))


check()
