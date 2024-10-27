from bs4 import BeautifulSoup


# Function to get open files
def get_soup(file_path):
    with open(file_path, encoding="utf-8") as f:
        html = f.read()
    soup = BeautifulSoup(html, "lxml")
    return soup


# Example div:
# <div class="bud-day">
#     <div class="bud-day-col">วันศุกร์ที่ 6 มกราคม 2566</div>
#     <div class="bud-day-col">ขึ้น ๑๕ ค่ำ เดือนยี่(๒) ปีขาล</div>
#     <div class="bud-day-col"></div>
# </div>


def Q1(file_path):
    soup = get_soup(file_path)
    # Create an empty dict to store the frequencies of each day
    bud_day_freq = {
        "วันจันทร์": 0,
        "วันอังคาร": 0,
        "วันพุธ": 0,
        "วันพฤหัสบดี": 0,
        "วันศุกร์": 0,
        "วันเสาร์": 0,
        "วันอาทิตย์": 0,
    }
    # Parent div
    bud_days = soup.find_all("div", class_="bud-day")

    # Iterate through the parent div
    for bud_day in bud_days:
        # Iterate through the child div
        day_infos = bud_day.find_all("div", class_="bud-day-col")
        for info in day_infos:
            # Add 1 to freq dict
            for day in bud_day_freq.keys():
                if day in info.get_text().strip():
                    bud_day_freq[day] += 1

    return list(bud_day_freq.values())


def Q2(file_path):
    soup = get_soup(file_path)
    bud_days = soup.find_all("div", class_="bud-day")

    for bud_day in bud_days:
        # Check if "วันวิสาขบูชา" is in the div and return if it's there
        if "วันวิสาขบูชา" in bud_day.get_text().strip():
            day_infos = bud_day.find_all("div", class_="bud-day-col")
            return day_infos[0].get_text().strip()


# Submission for Grader
# exec(input().strip())


# Testing
def check():
    file_path = "Grader 5/website.html"
    # https://dsde.nattee.net/problems/50/get_attachment
    print(Q1(file_path))
    print(Q2(file_path))


check()
