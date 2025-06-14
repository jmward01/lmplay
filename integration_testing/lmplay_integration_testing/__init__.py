"""
Integration testing package for lmplay framework.

This package provides end-to-end integration tests for the lmplay training
system, focusing on validating the complete training pipeline with real
model runners and mock data.

The tests are designed to:
- Validate end-to-end training functionality
- Ensure model saving/loading works correctly  
- Test different runner configurations
- Provide regression testing for core training logic
- Clean up artifacts after testing

Key modules:
- test_data: Mock training data generation
- test_utils: Testing utilities and cleanup helpers
- test_runners: End-to-end training tests for specific runners
- conftest: Pytest configuration and fixtures
"""