# 🤝 Contributing to Quantum AI Nexus

Thank you for your interest in contributing to Quantum AI Nexus! This document provides guidelines and instructions for contributing to this project.

## Table of Contents
1. [Code of Conduct](#code-of-conduct)
2. [Getting Started](#getting-started)
3. [Development Workflow](#development-workflow)
4. [Coding Standards](#coding-standards)
5. [Testing Guidelines](#testing-guidelines)
6. [Pull Request Process](#pull-request-process)
7. [Issue Reporting](#issue-reporting)

---

## 📜 Code of Conduct

### Our Pledge

We are committed to providing a welcoming and inspiring community for all. Please be respectful and constructive in your interactions.

### Expected Behavior

- ✅ Use welcoming and inclusive language
- ✅ Be respectful of differing viewpoints
- ✅ Accept constructive criticism gracefully
- ✅ Focus on what is best for the community
- ✅ Show empathy towards others

### Unacceptable Behavior

- ❌ Harassment or discriminatory language
- ❌ Trolling or insulting comments
- ❌ Personal or political attacks
- ❌ Public or private harassment
- ❌ Publishing others' private information

---

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- Git
- GitHub account
- Basic knowledge of Flask and AI/ML concepts

### Fork and Clone

```bash
# Fork the repository on GitHub
# Then clone your fork
git clone https://github.com/YOUR_USERNAME/quantum-ai-nexus.git
cd quantum-ai-nexus

# Add upstream remote
git remote add upstream https://github.com/MuhammadAbbas01/quantum-ai-nexus.git
```

### Set Up Development Environment

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
pip install -r requirements-dev.txt

# Set up pre-commit hooks
pip install pre-commit
pre-commit install
```

### Configure Environment

```bash
# Copy environment file
cp .env.example .env

# Edit .env with your API keys
nano .env
```

---

## 🔄 Development Workflow

### 1. Create a Branch

```bash
# Update main branch
git checkout main
git pull upstream main

# Create feature branch
git checkout -b feature/your-feature-name

# Or for bug fixes
git checkout -b fix/bug-description
```

### Branch Naming Convention

- `feature/` - New features
- `fix/` - Bug fixes
- `docs/` - Documentation updates
- `refactor/` - Code refactoring
- `test/` - Test additions or modifications
- `chore/` - Maintenance tasks

### 2. Make Changes

```bash
# Make your changes
# Test thoroughly
# Commit with meaningful messages
git add .
git commit -m "feat: add voice emotion detection"
```

### Commit Message Convention

Follow [Conventional Commits](https://www.conventionalcommits.org/):

```
<type>(<scope>): <description>

[optional body]

[optional footer]
```

**Types:**
- `feat` - New feature
- `fix` - Bug fix
- `docs` - Documentation
- `style` - Formatting
- `refactor` - Code restructuring
- `test` - Adding tests
- `chore` - Maintenance

**Examples:**
```bash
git commit -m "feat(voice): add emotion detection to speech processor"
git commit -m "fix(api): resolve session timeout issue"
git commit -m "docs: update API documentation for image endpoints"
```

### 3. Keep Your Branch Updated

```bash
# Fetch upstream changes
git fetch upstream

# Rebase your branch
git rebase upstream/main

# Or merge if you prefer
git merge upstream/main
```

---

## 💻 Coding Standards

### Python Style Guide

Follow [PEP 8](https://pep8.org/) with these specifics:

```python
# ✅ GOOD
def process_image(image_path: str, model_type: str = 'resnet') -> dict:
    """
    Process an image using specified model.
    
    Args:
        image_path: Path to the image file
        model_type: Type of model to use (default: 'resnet')
    
    Returns:
        dict: Analysis results containing objects, scene, etc.
    
    Raises:
        FileNotFoundError: If image file doesn't exist
        ValueError: If model_type is invalid
    """
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"Image not found: {image_path}")
    
    # Process image
    result = model.analyze(image_path)
    return result


# ❌ BAD
def processImage(imgPath):
    # No docstring, unclear naming, no type hints
    result=model.analyze(imgPath)
    return result
```

### Code Formatting

```bash
# Format code with Black
black .

# Sort imports
isort .

# Check style
flake8 .

# Type checking
mypy .
```

### Documentation Standards

```python
class ImageProcessor:
    """
    Image processing and analysis using deep learning models.
    
    This class provides comprehensive image analysis including object
    detection, scene classification, OCR, and image enhancement.
    
    Attributes:
        model_type (str): Type of model being used
        device (str): Computation device ('cpu' or 'cuda')
    
    Example:
        >>> processor = ImageProcessor(model_type='resnet')
        >>> result = processor.analyze('image.jpg')
        >>> print(result.objects)
        ['car', 'person', 'building']
    """
    
    def __init__(self, model_type: str = 'resnet'):
        """Initialize image processor with specified model."""
        self.model_type = model_type
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    def analyze(self, image_path: str) -> AnalysisResult:
        """
        Analyze an image comprehensively.
        
        Args:
            image_path: Path to image file
            
        Returns:
            AnalysisResult object containing detection results
            
        Raises:
            FileNotFoundError: If image file doesn't exist
        """
        pass
```

---

## 🧪 Testing Guidelines

### Write Tests for All New Features

```python
# tests/unit/test_image_processor.py

import pytest
from processors.image_processor import ImageProcessor

class TestImageProcessor:
    """Test suite for ImageProcessor class."""
    
    @pytest.fixture
    def processor(self):
        """Create ImageProcessor instance for testing."""
        return ImageProcessor(model_type='resnet')
    
    def test_analyze_valid_image(self, processor):
        """Test image analysis with valid image."""
        result = processor.analyze('tests/fixtures/test_image.jpg')
        
        assert result is not None
        assert len(result.objects) > 0
        assert result.confidence_score > 0.5
    
    def test_analyze_invalid_path(self, processor):
        """Test that FileNotFoundError is raised for invalid path."""
        with pytest.raises(FileNotFoundError):
            processor.analyze('nonexistent.jpg')
    
    def test_scene_classification(self, processor):
        """Test scene type classification."""
        result = processor.analyze('tests/fixtures/urban_scene.jpg')
        
        assert result.scene_type in ['urban', 'nature', 'indoor']
```

### Run Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=. --cov-report=html

# Run specific test file
pytest tests/unit/test_image_processor.py

# Run specific test
pytest tests/unit/test_image_processor.py::TestImageProcessor::test_analyze_valid_image
```

### Test Coverage Requirements

- Aim for **80%+ code coverage**
- All new features must include tests
- Bug fixes must include regression tests

---

## 📥 Pull Request Process

### Before Submitting

✅ **Checklist:**

- [ ] Code follows project style guidelines
- [ ] All tests pass (`pytest`)
- [ ] Code is properly formatted (`black`, `isort`)
- [ ] Documentation is updated
- [ ] Commit messages follow convention
- [ ] Branch is up to date with main
- [ ] No merge conflicts

### Submit Pull Request

1. **Push your branch**
   ```bash
   git push origin feature/your-feature-name
   ```

2. **Create Pull Request on GitHub**
   - Navigate to your fork on GitHub
   - Click "New Pull Request"
   - Select your feature branch
   - Fill in the PR template

### Pull Request Template

```markdown
## Description
Brief description of changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
Describe testing performed

## Checklist
- [ ] Code follows style guidelines
- [ ] Tests pass
- [ ] Documentation updated
- [ ] No breaking changes (or documented)

## Screenshots (if applicable)
Add screenshots here

## Related Issues
Fixes #(issue number)
```

### Review Process

1. **Automated Checks**
   - CI/CD pipeline runs tests
   - Code coverage is checked
   - Style guidelines are verified

2. **Code Review**
   - Maintainers review your code
   - Address any feedback
   - Make requested changes

3. **Merge**
   - Once approved, PR will be merged
   - Your contribution is live! 🎉

---

## 🐛 Issue Reporting

### Before Creating an Issue

- Search existing issues to avoid duplicates
- Verify the issue is reproducible
- Gather relevant information

### Bug Report Template

```markdown
## Bug Description
Clear description of the bug

## Steps to Reproduce
1. Go to '...'
2. Click on '...'
3. See error

## Expected Behavior
What should happen

## Actual Behavior
What actually happens

## Environment
- OS: [e.g., Ubuntu 22.04]
- Python Version: [e.g., 3.9.7]
- Browser: [if applicable]

## Screenshots
Add screenshots if applicable

## Additional Context
Any other relevant information
```

### Feature Request Template

```markdown
## Feature Description
Clear description of proposed feature

## Problem it Solves
What problem does this solve?

## Proposed Solution
How should this work?

## Alternatives Considered
Other solutions you've considered

## Additional Context
Any other relevant information
```

---

## 🎯 Areas for Contribution

### High Priority

- 🔴 **Performance Optimization**
  - Improve model inference speed
  - Optimize database queries
  - Reduce memory usage

- 🔴 **Testing**
  - Increase test coverage
  - Add integration tests
  - Performance benchmarking

- 🔴 **Documentation**
  - API examples
  - Tutorial videos
  - Architecture diagrams

### Medium Priority

- 🟡 **New Features**
  - Additional AI models
  - New API endpoints
  - Enhanced UI/UX

- 🟡 **Bug Fixes**
  - Fix reported issues
  - Improve error handling
  - Edge case handling

### Low Priority

- 🟢 **Enhancements**
  - Code refactoring
  - Dependency updates
  - Minor improvements

---

## 📚 Additional Resources

### Learning Materials

- [Flask Documentation](https://flask.palletsprojects.com/)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Transformers Library](https://huggingface.co/docs/transformers/)
- [OpenCV Documentation](https://docs.opencv.org/)

### Project-Specific Resources

- [Architecture Documentation](ARCHITECTURE.md)
- [API Documentation](API.md)
- [Deployment Guide](DEPLOYMENT.md)

---

## 💬 Communication

### GitHub Discussions

Use GitHub Discussions for:
- Questions about the project
- Feature discussions
- General conversation

**Repository:** https://github.com/MuhammadAbbas01/quantum-ai-nexus

### Issue Tracker

Use Issues for:
- Bug reports
- Feature requests
- Task tracking

---

## 🙏 Recognition

Contributors will be recognized in:
- README.md Contributors section
- Release notes
- Annual contributor report

---

## 📄 License

By contributing, you agree that your contributions will be licensed under the MIT License.

---

## 🎉 Thank You!

Thank you for contributing to Quantum AI Nexus! Your efforts help make this project better for everyone.

**Questions?** Feel free to reach out:
- Email: abbaskhan0011ehe@gmail.com
- GitHub: @MuhammadAbbas01
- LinkedIn: [linkedin.com/in/muhammadabbas-ai](https://www.linkedin.com/in/muhammadabbas-ai/)

---

**Last Updated**: January 2024  
**Maintainers**: Muhammad Abbas and contributors
