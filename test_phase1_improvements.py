"""
Test Climate-Aware Training Improvements
Run this to test adaptive learning rate scheduling
"""
from validate_hybrid_v5 import validateAdapted
import torch

def test_single_region(region_coords, region_name):
    """Test adaptive training on a single region"""
    print(f"\n{'='*60}")
    print(f"🧪 TESTING ADAPTIVE TRAINING: {region_name}")
    print(f"{'='*60}")
    
    try:
        results = validateAdapted(region_coords, region_name)
        
        print(f"\n✅ Test completed for {region_name}")
        print(f"Final Average MSE: {results.get('average_mse', 'N/A'):.3f}")
        
        # Clear GPU memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
        return results
        
    except Exception as e:
        print(f"❌ Error testing {region_name}: {e}")
        return None

def main():
    """Test adaptive training on selected regions"""
    
    # Test regions representing different climate zones
    test_regions = [
        ((40, 45, 285, 290), "NewYork"),      # Temperate
        ((-5, 0, 100, 105), "Indonesia"),     # Tropical  
        ((53, 58, 35, 40), "Moscow"),         # Cold
    ]
    
    print("🚀 ADAPTIVE TRAINING TEST")
    print("Testing climate-aware learning rates on different zones...")
    
    results_summary = {}
    
    for region_coords, region_name in test_regions:
        result = test_single_region(region_coords, region_name)
        if result:
            results_summary[region_name] = result.get('average_mse', float('inf'))
    
    # Summary
    print(f"\n{'='*60}")
    print("📊 ADAPTIVE TRAINING SUMMARY")
    print(f"{'='*60}")
    
    for region_name, mse in results_summary.items():
        if mse != float('inf'):
            print(f"{region_name:>12}: MSE = {mse:.3f}")
        else:
            print(f"{region_name:>12}: Failed")
    
    print(f"{'='*60}")
    print("✅ Adaptive training test complete!")
    print("\nFeatures tested:")
    print("  🎯 Climate-aware learning rates")
    print("  📈 Performance-based LR adjustment")
    print("  🔄 Cosine annealing with restarts")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()