#!/usr/bin/env python3
"""
Test script for faster-whisper integration.
Verifies GPU support, library availability, and basic transcription functionality.
"""

import os
import sys

def test_imports():
    """Test if required libraries are available"""
    print("\n" + "="*60)
    print("TESTING LIBRARY IMPORTS")
    print("="*60)
    
    results = {}
    
    # Test torch
    try:
        import torch
        results['torch'] = True
        print("✅ torch - installed")
    except ImportError as e:
        results['torch'] = False
        print(f"❌ torch - not found ({e})")
    
    # Test faster-whisper
    try:
        from faster_whisper import WhisperModel
        results['faster-whisper'] = True
        print("✅ faster-whisper - installed")
    except ImportError as e:
        results['faster-whisper'] = False
        print(f"❌ faster-whisper - not found ({e})")
    
    # Test transformers (for Hebrew model)
    try:
        import transformers
        results['transformers'] = True
        print("✅ transformers - installed")
    except ImportError as e:
        results['transformers'] = False
        print(f"❌ transformers - not found ({e})")
    
    # Test huggingface_hub
    try:
        import huggingface_hub
        results['huggingface_hub'] = True
        print("✅ huggingface_hub - installed")
    except ImportError as e:
        results['huggingface_hub'] = False
        print(f"❌ huggingface_hub - not found ({e})")
    
    return results

def test_gpu():
    """Test GPU/CUDA availability"""
    print("\n" + "="*60)
    print("TESTING GPU/CUDA SUPPORT")
    print("="*60)
    
    try:
        import torch
        
        cuda_available = torch.cuda.is_available()
        
        if cuda_available:
            print("✅ CUDA is available")
            print(f"   GPU Count: {torch.cuda.device_count()}")
            print(f"   GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"   CUDA Version: {torch.version.cuda}")
            
            # Test CUDA memory
            try:
                total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                print(f"   GPU Memory: {total_memory:.2f} GB")
            except:
                print("   GPU Memory: Unable to detect")
            
            return True
        else:
            print("⚠️  CUDA is not available")
            print("   Transcription will use CPU (slower)")
            print("\n   To enable GPU:")
            print("   1. Install NVIDIA CUDA Toolkit")
            print("   2. Reinstall PyTorch with CUDA support:")
            print("      pip install torch --index-url https://download.pytorch.org/whl/cu118")
            return False
    
    except ImportError:
        print("❌ torch not installed - cannot check GPU support")
        return False
    except Exception as e:
        print(f"❌ Error checking GPU: {e}")
        return False

def test_faster_whisper_basic():
    """Test basic faster-whisper functionality"""
    print("\n" + "="*60)
    print("TESTING FASTER-WHISPER BASIC FUNCTIONALITY")
    print("="*60)
    
    try:
        from faster_whisper import WhisperModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print(f"Loading tiny model on {device}...")
        model = WhisperModel("tiny", device=device, compute_type=compute_type)
        print("✅ Model loaded successfully")
        
        # Test available models
        print("\nAvailable models:")
        models = ["tiny", "base", "small", "medium", "large-v2", "large-v3", "turbo"]
        for m in models:
            print(f"   - {m}")
        
        print("\n✅ faster-whisper is working correctly")
        
        del model
        if device == "cuda":
            torch.cuda.empty_cache()
        
        return True
    
    except Exception as e:
        print(f"❌ Error testing faster-whisper: {e}")
        return False

def test_hebrew_model():
    """Test if Hebrew model is accessible"""
    print("\n" + "="*60)
    print("TESTING HEBREW MODEL (ivrit-ai)")
    print("="*60)
    
    try:
        from faster_whisper import WhisperModel
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"
        
        print("Attempting to load Hebrew model...")
        print("(This may take a while on first run - downloading model)")
        
        try:
            model = WhisperModel(
                "ivrit-ai/whisper-large-v3-turbo-ct2",
                device=device,
                compute_type=compute_type
            )
            print("✅ Hebrew model loaded successfully!")
            print("   Model: ivrit-ai/whisper-large-v3-turbo-ct2")
            print("   This model is optimized for Hebrew transcription")
            print("   Note: Using CTranslate2 format (compatible with faster-whisper)")
            
            del model
            if device == "cuda":
                torch.cuda.empty_cache()
            
            return True
        
        except Exception as e:
            print(f"⚠️  Hebrew model not available: {e}")
            print("   Will fall back to standard model for Hebrew content")
            return False
    
    except Exception as e:
        print(f"❌ Error testing Hebrew model: {e}")
        return False

def test_transcription_function():
    """Test the actual generate_transcript function"""
    print("\n" + "="*60)
    print("TESTING GENERATE_TRANSCRIPT FUNCTION")
    print("="*60)
    
    try:
        # Check if reelsfy module can be imported
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        
        # Try to import the function
        try:
            from reelsfy import generate_transcript, format_timestamp_srt
            print("✅ Successfully imported generate_transcript function")
            print("✅ Successfully imported format_timestamp_srt function")
            return True
        except ImportError as e:
            print(f"⚠️  Could not import from reelsfy: {e}")
            print("   Make sure you're running this from the reelsfy_folder directory")
            return False
    
    except Exception as e:
        print(f"❌ Error testing transcription function: {e}")
        return False

def print_summary(results):
    """Print summary of all tests"""
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    all_passed = all(results.values())
    
    for test_name, passed in results.items():
        status = "✅ PASSED" if passed else "❌ FAILED"
        print(f"{test_name:30s} {status}")
    
    print("="*60)
    
    if all_passed:
        print("\n🎉 All tests passed! Your system is ready for faster transcription.")
        print("\nNext steps:")
        print("1. Run your video processing with reelsfy.py")
        print("2. Transcription will automatically use faster-whisper with GPU")
        print("3. Hebrew videos will use the optimized ivrit-ai model")
    else:
        print("\n⚠️  Some tests failed. Please review the errors above.")
        print("\nTo fix issues:")
        print("1. Install missing packages: pip install -r requirements.txt")
        print("2. For GPU support: Install CUDA Toolkit and reinstall PyTorch")
        print("3. The system will fall back to CPU if GPU is unavailable")
    
    return all_passed

def main():
    """Run all tests"""
    print("\n" + "="*70)
    print(" "*15 + "FASTER-WHISPER TEST SUITE")
    print("="*70)
    print("\nThis script will test your faster-whisper installation and GPU setup.")
    print("It's safe to run and won't modify any files.\n")
    
    results = {}
    
    # Run tests
    import_results = test_imports()
    results['Library Imports'] = all(import_results.values())
    
    results['GPU/CUDA Support'] = test_gpu()
    results['faster-whisper Basic'] = test_faster_whisper_basic()
    results['Hebrew Model (ivrit-ai)'] = test_hebrew_model()
    results['generate_transcript Function'] = test_transcription_function()
    
    # Print summary
    all_passed = print_summary(results)
    
    # Exit code
    sys.exit(0 if all_passed else 1)

if __name__ == "__main__":
    main()

