from adapt_hybrid_v5 import adaptModel
from validate_hybrid_v5 import validateAdapted
import time
import os
import torch

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
    ((18, 23, 75, 80), "India"),
]

time_taken = {}

for region in regions:
    try:
        start = time.time()
        region_coords, region_name = region
        print(f"\n🌍 Processing region: {region_name}")
        
        # Check if adapted model exists, if not, adapt first
        adapted_path = f"./Out_Data/AdaptedModels/hybrid_v5_adapted_{region_name}_{region_coords}.pt"
        
        if not os.path.exists(adapted_path):
            print(f"🔄 Adapting Model V5 for {region_name}...")
            adaptModel(region_coords, region_name)
        else:
            print(f"✅ Using existing adapted model for {region_name}")
        
        # Validate the adapted model
        print(f"🎯 Validating {region_name}...")
        validateAdapted(region_coords, region_name)
        print(f"✅ Completed processing for {region_name}")
        
        end = time.time()
        time_taken[region_name] = end - start
        print(f"⏱️ Time taken for {region_name}: {time_taken[region_name]:.1f}s")
        
        # Clear GPU memory after each region
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"❌ Error processing {region_name}: {e}")
        # Clear GPU memory on error too
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

print("\n" + "=" * 60)
print("📊 MODEL V5 PROCESSING SUMMARY")
print("=" * 60)
for region_name, duration in time_taken.items():
    print(f"{region_name:>15}: {duration/60:.1f} min")
print("=" * 60)
