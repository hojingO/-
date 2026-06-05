import csv

# We are targeting suncheon.csv
file_path = "suncheon.csv"

# Read existing data
with open(file_path, mode='r', encoding='utf-8') as f:
    reader = csv.reader(f)
    headers = next(reader)
    rows = list(reader)

# Check if already added
already_added = any(row[1] == "순천만국가정원" for row in rows)

if not already_added:
    garden_row = [
        "46150-99999",                     # id
        "순천만국가정원",                    # park_name
        "국가정원",                          # park_type
        "전라남도 순천시 국가정원1호길 162-2", # address_road
        "전라남도 순천시 오천동 600",         # address_land
        "34.929837",                        # latitude
        "127.502758",                       # longitude
        "1120000",                          # area
        "없음",                             # sports_facility
        "없음",                             # amusement_facility
        "식음시설",                          # convenience_facility
        "생태학습관",                        # cultural_facility
        "조경시설",                          # other_facility
        "2013-10-20",                       # designated_date
        "전라남도 순천시청",                 # management_agency
        "061-749-3114",                     # tel
        "2026-06-05",                       # data_date
        "4840000",                          # agency_code
        "전라남도 순천시"                    # agency_name
    ]
    
    if len(garden_row) < len(headers):
        garden_row += [""] * (len(headers) - len(garden_row))
    elif len(garden_row) > len(headers):
        garden_row = garden_row[:len(headers)]
        
    rows.append(garden_row)
    
    with open(file_path, mode='w', encoding='utf-8', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(headers)
        writer.writerows(rows)
    print("Successfully added Suncheonman National Garden to suncheon.csv!")
else:
    print("Suncheonman National Garden is already in suncheon.csv.")
