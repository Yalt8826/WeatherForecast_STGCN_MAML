from adapt_hybrid_v4 import adaptModel
import validate_hybrid_v4 as validate
from validate_hybrid_v4 import validateAdapted
import time

regions = [
    ((40, 45, 285, 290), "NewYork"),
    ((-5, 0, 100, 105), "Indonesia"),
    ((53, 58, 35, 40), "Moscow"),
    ((8, 13, 98, 103), "Thailand"),
    ((-33, -28, -70, -65), "Argentina"),
    ((-17, -12, 145, 150), "QueensAustralia"),
    ((70, 75, 82, 87), "NorthSiberia"),
    ((35, 40, 69, 74), "Afghanistan"),
    ((15, 20, 30, 35), "Sudan"),
]

time_taken = {}

for region in regions:
    try:
        start = time.time()
        region_coords, region_name = region
        print(f"\n🌍 Processing region: {region_name}")
        adaptModel(region_coords, region_name)
        validateAdapted(region_coords, region_name)
        print(f"✅ Completed validation for {region_name}")
        end = time.time()
        time_taken[region_name] = end - start
        print(f"⏱️ Time taken for {region_name}: {time_taken[region_name]}")

    except Exception as e:
        print(f"❌ Error processing {region_name}: {e}")

print("\nTime taken for each region:")
for region_name, duration in time_taken.items():
    print(f"{region_name}: {duration/3600:.2f} hrs")
