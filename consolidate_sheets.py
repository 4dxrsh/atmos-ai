import gspread
import pandas as pd
import os
import time

print("--- Phase 1: Consolidate Data ---")

# --- AUTHENTICATION ---
try:
    gc = gspread.service_account(filename='service-account.json')
    print("Authentication successful.")
except Exception as e:
    print(f"Error: {e}. Make sure 'service-account.json' is in the folder and shared with your sheets.")
    exit()

# === PART 1: PROCESS STATION BOOKS ===

# TODO: FILL THIS OUT
# You must provide the lat/lon for each station.
STATION_BOOK_LINKS = {
    "https://docs.google.com/spreadsheets/d/1vfXI82bNxR9p4ka7xc5_0ZzU8lOFtpppPz8phQTovRU/edit?usp=sharing": {"lat": 12.7791, "lon": 77.6436},
    "https://docs.google.com/spreadsheets/d/1AR1NA132RodYpCUTYPbnO2O71nLJH-OUUUzh8jP6V0k/edit?usp=sharing": {"lat": 12.9912, "lon": 77.5439},
    "https://docs.google.com/spreadsheets/d/1chVVRw0q2LzGIbCcJB5MvsFHSYdkf6uqj_Rzsr9hAoo/edit?usp=sharing": {"lat": 12.9166, "lon": 77.6101},
    "https://docs.google.com/spreadsheets/d/1ozvaV_2bYElQ9Kw3tdJm-7QSEr_YGN-92oHDDFl6L1U/edit?usp=sharing": {"lat": 13.0061, "lon": 77.6594},
    "https://docs.google.com/spreadsheets/d/1pGuEl4FzLpz9J60AbsCcTdsOa4pwERlMo7nt9sW4FRE/edit?usp=sharing": {"lat": 12.9353, "lon": 77.6696},
    "https://docs.google.com/spreadsheets/d/1GQExwVK8cmZLe_inRzg46MbgoVVGDQLWqzUiTmEWfaQ/edit?usp=sharing": {"lat": 12.9177, "lon": 77.6238},
    "https://docs.google.com/spreadsheets/d/1ozLInGUeRuxew0J0VzH7WVV1UIlAk0h2IFFrL75DZtc/edit?usp=sharing": {"lat": 12.9781, "lon": 77.5697},
    "https://docs.google.com/spreadsheets/d/1RsgJhAqnFntoBCWXR82rTks4vbpr8NANqEPiXF5QmFE/edit?usp=sharing": {"lat": 13.0285, "lon": 77.5197},
    "https://docs.google.com/spreadsheets/d/1JWvpsw3x_nSogMedIcoq-8M9B1V1Au8lBqTh40w0YV8/edit?usp=sharing": {"lat": 13.0354, "lon": 77.5988},
    "https://docs.google.com/spreadsheets/d/1FIDUWSz2wpy0LMpjTbvLzGcx5V3Sc6u9LXZyQE1GgEk/edit?usp=sharing": {"lat": 12.9866, "lon": 77.4904},
    "https://docs.google.com/spreadsheets/d/1fSlvYX76dTfVjcra98dttTZl5c4LXo9_GpIfmOgJ9dA/edit?usp=sharing": {"lat": 12.9375, "lon": 77.5949},
    "https://docs.google.com/spreadsheets/d/10bCLeVYCSAr_iFD0LRkdeNTdiX73Y_wuKuK2w9WxAa8/edit?usp=sharing": {"lat": 12.9238, "lon": 77.4985},
    "https://docs.google.com/spreadsheets/d/1_vuWk9rVmH2suLXU44p6zT9zQDbKEDWCENG2rtoLjBE/edit?usp=sharing": {"lat": 12.9199, "lon": 77.5837},
    "https://docs.google.com/spreadsheets/d/1OREAJTp422hLd5921BkahxuVQEY_P0jqr2l8by8h67A/edit?usp=sharing": {"lat": 12.9568, "lon": 77.5397},
    
    
}

all_station_data = []
print(f"Processing {len(STATION_BOOK_LINKS)} station book(s)...")

for link, info in STATION_BOOK_LINKS.items():
    try:
        book = gc.open_by_url(link)
        print(f"  Opening book: {book.title}")
        
        for sheet in book.worksheets():
            year_str = sheet.title
            if not year_str.isdigit() or len(year_str) != 4:
                print(f"    ... skipping non-year sheet: '{sheet.title}'")
                continue
                
            print(f"    ... processing year: {year_str}")
            all_values = sheet.get_all_values()
            header = all_values[0]
            data_rows = all_values[1:]
            
            try:
                day_col_index = header.index("Day")
            except ValueError:
                print("    ... 'Day' column not found. Skipping sheet.")
                continue
                
            for month_idx in range(day_col_index + 1, len(header)):
                month_name = header[month_idx]
                if not month_name: continue
                
                daily_aqi_values = []
                for row in data_rows:
                    if len(row) > month_idx and row[day_col_index].isdigit():
                        aqi_str = row[month_idx]
                        if aqi_str:
                            try:
                                daily_aqi_values.append(float(aqi_str))
                            except ValueError:
                                pass 
                
                if daily_aqi_values:
                    avg_aqi = sum(daily_aqi_values) / len(daily_aqi_values)
                    timestamp = pd.to_datetime(f"{month_name} {year_str}")
                    all_station_data.append({
                        'timestamp': timestamp,
                        'latitude': info['lat'],
                        'longitude': info['lon'],
                        'station_aqi': avg_aqi
                    })

    except Exception as e:
        print(f"    Failed to process {link}: {e}")
    print("    ... Pausing for 10 seconds to respect rate limit ...")
    time.sleep(10)
    
if not all_station_data:
    print("No station data found. Exiting.")
    exit()

df_stations = pd.DataFrame(all_station_data)
df_stations.to_csv('./data/station_ground_truth.csv', index=False)
print(f"Successfully saved {len(df_stations)} station records.")


# TODO: FILL THIS OUT
CITY_BOOK_LINK = "https://docs.google.com/spreadsheets/d/16asVI_PSQz3yC_VItGyHkmo-XqVaWLwrNECuIvFV_34/edit?usp=sharing"

all_city_data = []
print(f"\nProcessing City Baseline Book...")

try:
    if "YOUR_CITY_BOOK_LINK" in CITY_BOOK_LINK:
         print("City Baseline Book link not set. Skipping Part 2.")
    else:
        book = gc.open_by_url(CITY_BOOK_LINK)
        print(f"  Opening city book: {book.title}")

        # This logic is now identical to Part 1
        for sheet in book.worksheets():
            year_str = sheet.title
            # This check skips "23p", "24p", etc.
            if not year_str.isdigit() or len(year_str) != 4:
                print(f"    ... skipping non-year sheet: '{sheet.title}'")
                continue
                
            print(f"    ... processing year: {year_str}")
            all_values = sheet.get_all_values()
            
            if not all_values:
                print("    ... sheet is empty.")
                continue

            header = all_values[0]
            data_rows = all_values[1:]
            
            try:
                day_col_index = header.index("Day")
            except ValueError:
                print("    ... 'Day' column not found. Skipping sheet.")
                continue
                
            for month_idx in range(day_col_index + 1, len(header)):
                month_name = header[month_idx]
                if not month_name: continue
                
                daily_aqi_values = []
                for row in data_rows:
                    if len(row) > month_idx and row[day_col_index].isdigit():
                        aqi_str = row[month_idx]
                        if aqi_str:
                            try:
                                daily_aqi_values.append(float(aqi_str))
                            except ValueError:
                                pass 
                
                if daily_aqi_values:
                    avg_aqi = sum(daily_aqi_values) / len(daily_aqi_values)
                    timestamp = pd.to_datetime(f"{month_name} {year_str}")
                    all_city_data.append({
                        'timestamp': timestamp,
                        'baseline_aqi': avg_aqi
                    })

except Exception as e:
    print(f"    Failed to process city book: {e}")

if not all_city_data:
    print("No city data found. 'city_baseline_data.csv' not created.")
else:
    df_city = pd.DataFrame(all_city_data)
    df_city.to_csv('./data/city_baseline_data.csv', index=False)
    print(f"Successfully saved {len(df_city)} city records to ./data/city_baseline_data.csv")


# ==================================
# === PART 3: FINAL MERGE ===
# ==================================
print("\nMerging all data...")
try:
    df_stations = pd.read_csv("./data/station_ground_truth.csv", parse_dates=["timestamp"])
    df_city = pd.read_csv("./data/city_baseline_data.csv", parse_dates=["timestamp"])

    df_master = pd.merge(df_stations, df_city, on="timestamp")
    df_master['month'] = df_master['timestamp'].dt.month
    
    df_master.to_csv("./data/final_training_data.csv", index=False)
    print("Success! Created 'final_training_data.csv'.")
    print("You are ready for the next step.")

except FileNotFoundError:
    print("\nError: Could not create 'final_training_data.csv'.")
    print("One of the source files ('station_ground_truth.csv' or 'city_baseline_data.csv') is missing.")
    print("Please check the TODOs in Part 1 and Part 2 and re-run.")
except Exception as e:
    print(f"Error merging files: {e}")
