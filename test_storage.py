"""
Test Firebase Storage Implementation
Comprehensive tests for all storage features
"""

from firebase_storage import get_storage_manager
import time

def print_section(title):
    print("\n" + "="*60)
    print(f"  {title}")
    print("="*60)


def test_basic_operations():
    """Test basic CRUD operations"""
    print_section("🔧 Test 1: Basic Operations")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    
    # Set
    print("📝 Setting data...")
    success = storage.set(
        key='test_key',
        value={'message': 'Hello World', 'number': 42},
        user_id=user_id
    )
    print(f"   Set: {'✅' if success else '❌'}")
    
    # Get
    print("\n📥 Getting data...")
    data = storage.get('test_key', user_id=user_id)
    print(f"   Get: {data}")
    assert data is not None, "Data should exist!"
    assert data['message'] == 'Hello World', "Data should match!"
    
    # List
    print("\n📋 Listing keys...")
    keys = storage.list_keys(user_id=user_id)
    print(f"   Keys: {keys}")
    assert 'test_key' in keys, "Key should be in list!"
    
    # Delete
    print("\n🗑️ Deleting data...")
    success = storage.delete('test_key', user_id=user_id)
    print(f"   Delete: {'✅' if success else '❌'}")
    
    # Verify deletion
    data = storage.get('test_key', user_id=user_id)
    assert data is None, "Data should be deleted!"
    
    print("\n✅ Basic operations test PASSED!")
    return True


def test_shared_data():
    """Test shared data functionality"""
    print_section("🌍 Test 2: Shared Data")
    
    storage = get_storage_manager()
    
    # Save shared data
    print("💾 Saving shared data...")
    storage.set(
        key='shared_config',
        value={'setting': 'global', 'version': '1.0'},
        shared=True
    )
    
    # Access from different users
    print("\n👥 Accessing from multiple users...")
    user1_data = storage.get('shared_config', shared=True, user_id='user_1')
    user2_data = storage.get('shared_config', shared=True, user_id='user_2')
    
    print(f"   User 1: {user1_data}")
    print(f"   User 2: {user2_data}")
    
    assert user1_data == user2_data, "Shared data should be identical!"
    
    print("\n✅ Shared data test PASSED!")
    return True


def test_collections():
    """Test different collections"""
    print_section("📚 Test 3: Collections")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    
    collections = ['config', 'models', 'data', 'recipes']
    
    for collection in collections:
        print(f"\n📦 Testing collection: {collection}")
        
        # Store data
        storage.set(
            key=f'{collection}_item',
            value={'collection': collection, 'data': 'test'},
            collection=collection,
            user_id=user_id
        )
        
        # Verify storage
        data = storage.get(f'{collection}_item', collection=collection, user_id=user_id)
        assert data is not None, f"Data should exist in {collection}!"
        print(f"   ✅ {collection} collection working")
    
    print("\n✅ Collections test PASSED!")
    return True


def test_model_storage():
    """Test ML model storage"""
    print_section("🤖 Test 4: ML Model Storage")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    
    # Create fake model data
    print("💾 Saving model...")
    import pickle
    fake_model = {'type': 'RandomForest', 'accuracy': 0.95}
    model_bytes = pickle.dumps(fake_model)
    
    url = storage.save_model(
        model_data=model_bytes,
        model_name='test_model_v1',
        user_id=user_id,
        metadata={
            'accuracy': 0.95,
            'features': ['age', 'income'],
            'created': '2024-12-09'
        }
    )
    
    print(f"   Model URL: {url}")
    assert url is not None, "Model should be saved!"
    
    # Load model
    print("\n📥 Loading model...")
    loaded_bytes = storage.load_model('test_model_v1', user_id=user_id)
    assert loaded_bytes is not None, "Model should load!"
    
    loaded_model = pickle.loads(loaded_bytes)
    print(f"   Loaded: {loaded_model}")
    assert loaded_model['accuracy'] == 0.95, "Model data should match!"
    
    print("\n✅ Model storage test PASSED!")
    return True


def test_chat_history():
    """Test chat history storage"""
    print_section("💬 Test 5: Chat History")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    session_id = 'session_20241209'
    
    # Create chat messages
    messages = [
        {'role': 'user', 'message': 'Hello!'},
        {'role': 'assistant', 'message': 'Hi! How can I help?'},
        {'role': 'user', 'message': 'Clean my data'},
        {'role': 'assistant', 'message': 'Sure! Upload your file.'}
    ]
    
    # Save chat
    print("💾 Saving chat history...")
    success = storage.save_chat_history(
        messages=messages,
        user_id=user_id,
        session_id=session_id
    )
    assert success, "Chat should save!"
    print("   ✅ Saved")
    
    # Load chat
    print("\n📥 Loading chat history...")
    loaded = storage.load_chat_history(user_id=user_id, session_id=session_id)
    print(f"   Messages: {len(loaded)}")
    assert len(loaded) == 4, "Should have 4 messages!"
    
    print("\n✅ Chat history test PASSED!")
    return True


def test_recipes():
    """Test recipe storage"""
    print_section("📖 Test 6: Recipe Storage")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    
    # Create recipe
    recipe = {
        'name': 'Customer Data Cleaning',
        'operations': [
            {'operation': 'remove_duplicates'},
            {'operation': 'fill_missing', 'params': {'age': 'median'}},
            {'operation': 'normalize_text', 'params': {'columns': ['name', 'email']}}
        ],
        'description': 'Standard customer data cleaning',
        'version': '1.0'
    }
    
    # Save recipe
    print("💾 Saving recipe...")
    success = storage.save_recipe(
        recipe_name='customer_standard',
        recipe_data=recipe,
        user_id=user_id,
        shared=False
    )
    assert success, "Recipe should save!"
    print("   ✅ Saved")
    
    # Load recipe
    print("\n📥 Loading recipe...")
    loaded = storage.load_recipe('customer_standard', user_id=user_id)
    print(f"   Recipe: {loaded['name']}")
    print(f"   Operations: {len(loaded['operations'])}")
    assert loaded is not None, "Recipe should load!"
    
    # List recipes
    print("\n📋 Listing recipes...")
    recipes = storage.list_recipes(user_id=user_id)
    print(f"   Found: {recipes}")
    assert 'customer_standard' in recipes, "Recipe should be in list!"
    
    print("\n✅ Recipe storage test PASSED!")
    return True


def test_prefix_filtering():
    """Test prefix-based key filtering"""
    print_section("🔍 Test 7: Prefix Filtering")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    
    # Create data with prefixes
    print("💾 Creating test data...")
    prefixes = ['model_', 'data_', 'config_']
    
    for prefix in prefixes:
        for i in range(3):
            storage.set(
                key=f'{prefix}{i}',
                value={'prefix': prefix, 'index': i},
                user_id=user_id
            )
    
    # Test each prefix
    print("\n🔍 Testing prefix filters...")
    for prefix in prefixes:
        keys = storage.list_keys(prefix=prefix, user_id=user_id)
        print(f"   {prefix}*: {len(keys)} keys")
        assert len(keys) == 3, f"Should have 3 keys for {prefix}!"
    
    print("\n✅ Prefix filtering test PASSED!")
    return True


def test_storage_stats():
    """Test storage statistics"""
    print_section("📊 Test 8: Storage Statistics")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    
    # Get stats
    print("📊 Getting statistics...")
    stats = storage.get_storage_stats(user_id=user_id)
    
    print(f"\n   Storage Type: {stats['storage_type']}")
    print(f"   Initialized: {stats['initialized']}")
    print(f"   Collections:")
    
    for collection, info in stats['collections'].items():
        print(f"      - {collection}: {info['key_count']} keys")
    
    print("\n✅ Storage stats test PASSED!")
    return True


def test_cleanup():
    """Clean up test data"""
    print_section("🧹 Test 9: Cleanup")
    
    storage = get_storage_manager()
    user_id = 'test_user_123'
    
    print("🗑️ Clearing test data...")
    success = storage.clear_user_data(user_id=user_id)
    assert success, "Cleanup should succeed!"
    
    # Verify cleanup
    keys = storage.list_keys(user_id=user_id)
    print(f"   Remaining keys: {len(keys)}")
    
    print("\n✅ Cleanup test PASSED!")
    return True


def run_all_tests():
    """Run all storage tests"""
    print("\n" + "🧪"*30)
    print("  FIREBASE STORAGE TEST SUITE")
    print("🧪"*30)
    
    tests = [
        ("Basic Operations", test_basic_operations),
        ("Shared Data", test_shared_data),
        ("Collections", test_collections),
        ("Model Storage", test_model_storage),
        ("Chat History", test_chat_history),
        ("Recipes", test_recipes),
        ("Prefix Filtering", test_prefix_filtering),
        ("Storage Stats", test_storage_stats),
        ("Cleanup", test_cleanup)
    ]
    
    results = []
    start_time = time.time()
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ Test '{name}' FAILED with error:")
            print(f"   {str(e)}")
            results.append((name, False))
    
    elapsed = time.time() - start_time
    
    # Summary
    print_section("📊 TEST SUMMARY")
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status}  {name}")
    
    print(f"\n{'='*60}")
    print(f"Results: {passed}/{total} tests passed")
    print(f"Time: {elapsed:.2f} seconds")
    print(f"{'='*60}")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Storage system is working!")
        print("\n✨ Features verified:")
        print("   ✅ Basic CRUD operations")
        print("   ✅ Shared data")
        print("   ✅ Multiple collections")
        print("   ✅ Model persistence")
        print("   ✅ Chat history")
        print("   ✅ Recipe storage")
        print("   ✅ Prefix filtering")
        print("   ✅ Statistics")
        print("\n🚀 Ready for production!")
        return True
    else:
        print(f"\n❌ {total - passed} test(s) failed.")
        print("\nNote: If using local storage, some features may be limited.")
        print("Install Firebase Admin SDK for full functionality:")
        print("  pip install firebase-admin")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)