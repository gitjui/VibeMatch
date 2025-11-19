"""
Test Client for Music Identification API
=========================================

Run this script to test all API endpoints.
Make sure the API is running first: ./start_api.sh
"""

import requests
import json
from pathlib import Path

BASE_URL = "http://localhost:8000"

def test_health():
    """Test health check endpoint"""
    print("\n" + "="*60)
    print("TEST 1: Health Check")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/health")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_stats():
    """Test statistics endpoint"""
    print("\n" + "="*60)
    print("TEST 2: Database Statistics")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/stats")
    print(f"Status: {response.status_code}")
    print(json.dumps(response.json(), indent=2))
    return response.status_code == 200


def test_list_songs():
    """Test list songs endpoint"""
    print("\n" + "="*60)
    print("TEST 3: List Songs")
    print("="*60)
    
    response = requests.get(f"{BASE_URL}/songs?limit=5")
    print(f"Status: {response.status_code}")
    data = response.json()
    print(f"Total songs: {data['pagination']['total']}")
    print(f"Showing: {len(data['songs'])} songs")
    
    for song in data['songs'][:3]:
        print(f"  - {song['title']} by {song['artist']}")
    
    return response.status_code == 200


def test_get_song_details():
    """Test get song details endpoint"""
    print("\n" + "="*60)
    print("TEST 4: Get Song Details")
    print("="*60)
    
    song_id = 1
    response = requests.get(f"{BASE_URL}/songs/{song_id}")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        song = response.json()
        print(f"Song: {song['title']} by {song['artist']}")
        print(f"Album: {song.get('album', 'N/A')}")
        print(f"Duration: {song.get('duration_seconds', 'N/A')}s")
        print(f"Tempo: {song.get('tempo_bpm', 'N/A')} BPM")
    
    return response.status_code == 200


def test_get_recommendations():
    """Test recommendations endpoint"""
    print("\n" + "="*60)
    print("TEST 5: Get Recommendations")
    print("="*60)
    
    song_id = 1
    response = requests.get(f"{BASE_URL}/recommendations/{song_id}?limit=5")
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        print(f"Original song: {data['song']['title']} by {data['song']['artist']}")
        print(f"\nRecommendations ({data['total']}):")
        
        for rec in data['recommendations'][:5]:
            print(f"  - {rec['title']} by {rec['artist']}")
    
    return response.status_code == 200


def test_identify_url():
    """Test identify from URL endpoint"""
    print("\n" + "="*60)
    print("TEST 6: Identify from File Path")
    print("="*60)
    
    # Use a song from your collection
    test_file = "/Users/juigupte/Desktop/Learning/music/mp3/Ed Sheeran - Shape of You (Official Music Video).mp3"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("Skipping this test")
        return False
    
    payload = {
        "file_path": test_file,
        "start_time": 30.0,
        "duration": 5.0
    }
    
    response = requests.post(f"{BASE_URL}/identify-url", json=payload)
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if data['exact_match']['found']:
            match = data['exact_match']
            print(f"\n✅ EXACT MATCH FOUND!")
            print(f"   Song: {match['title']} by {match['artist']}")
            print(f"   Confidence: {match['confidence']*100:.1f}%")
        else:
            print("\n❌ No exact match found")
        
        print(f"\n🎵 Recommendations: {data['summary']['total_recommendations']}")
        for rec in data['recommendations'][:5]:
            print(f"   {rec['rank']}. {rec['title']} - {rec['artist']} ({rec['similarity']*100:.0f}%)")
    
    return response.status_code == 200


def test_identify_upload():
    """Test identify from file upload endpoint"""
    print("\n" + "="*60)
    print("TEST 7: Identify from File Upload")
    print("="*60)
    
    test_file = "/Users/juigupte/Desktop/Learning/music/mp3/Dua Lipa - Be The One (Official Music Video) [-rey3m8SWQI].mp3"
    
    if not Path(test_file).exists():
        print(f"⚠️  Test file not found: {test_file}")
        print("Skipping this test")
        return False
    
    with open(test_file, 'rb') as f:
        files = {'file': (Path(test_file).name, f, 'audio/mpeg')}
        params = {'start_time': 30.0, 'duration': 5.0}
        
        response = requests.post(f"{BASE_URL}/identify", files=files, params=params)
    
    print(f"Status: {response.status_code}")
    
    if response.status_code == 200:
        data = response.json()
        
        if data['exact_match']['found']:
            match = data['exact_match']
            print(f"\n✅ EXACT MATCH FOUND!")
            print(f"   Song: {match['title']} by {match['artist']}")
            print(f"   Confidence: {match['confidence']*100:.1f}%")
        else:
            print("\n❌ No exact match found")
        
        print(f"\n🎵 Recommendations: {data['summary']['total_recommendations']}")
        for rec in data['recommendations'][:3]:
            print(f"   {rec['rank']}. {rec['title']} - {rec['artist']}")
    
    return response.status_code == 200


def main():
    """Run all tests"""
    print("="*60)
    print("Music Identification API - Test Suite")
    print("="*60)
    print(f"Testing API at: {BASE_URL}")
    
    # Check if API is running
    try:
        response = requests.get(f"{BASE_URL}/")
        print("✅ API is running!")
    except requests.exceptions.ConnectionError:
        print("\n❌ ERROR: API is not running!")
        print("\nPlease start the API first:")
        print("  ./start_api.sh")
        print("\nOr run:")
        print("  uvicorn api:app --reload --port 8000")
        return
    
    # Run tests
    results = []
    
    results.append(("Health Check", test_health()))
    results.append(("Database Stats", test_stats()))
    results.append(("List Songs", test_list_songs()))
    results.append(("Get Song Details", test_get_song_details()))
    results.append(("Get Recommendations", test_get_recommendations()))
    results.append(("Identify from URL", test_identify_url()))
    results.append(("Identify from Upload", test_identify_upload()))
    
    # Summary
    print("\n" + "="*60)
    print("TEST SUMMARY")
    print("="*60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {test_name}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n🎉 All tests passed!")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed")
    
    print("\n" + "="*60)
    print("API Documentation:")
    print(f"  Swagger UI: {BASE_URL}/docs")
    print(f"  ReDoc: {BASE_URL}/redoc")
    print("="*60)


if __name__ == "__main__":
    main()
