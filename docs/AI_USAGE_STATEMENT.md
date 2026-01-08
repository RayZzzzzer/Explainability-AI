# Generative AI Usage Statement

**Project**: Unified XAI Platform  
**Team**: [Your TD Group Number]  
**Date**: December 2025

---

## Declaration

This document serves as a transparent declaration of how generative AI tools were used in the development of this project, in accordance with academic integrity guidelines.

---

## 1. Tools Used

### Primary Tool: GitHub Copilot
- **Model**: Claude Sonnet 4.5
- **Interface**: Visual Studio Code
- **License**: [Educational/Personal/Professional]
- **Usage Period**: December 2025 - January 2026

### Secondary Tools
- ChatGPT / other LLMs for prompting or specific debugging matters that did not require too much context.
- Automatic autocompletion in Visual Studio Code.

---

## 2. Scope of AI Usage

### 2.1 Code Generation

#### What AI Did:
- Generated "template" code for Python modules
- Created class structures with methods and docstrings
- Implemented standard design patterns (e.g., factory pattern for model loading)
- Wrote utility functions (preprocessing, file handling)
- Generated Streamlit UI components
- Debugged issues encountered along the way

#### Specific Examples:
1. **Preprocessing Module**: AI generated the skeleton for `AudioPreprocessor` and `ImagePreprocessor` classes
2. **XAI Wrappers**: AI created initial implementations of LIME, Grad-CAM, and SHAP explainer classes
3. **UI Layout**: AI generated Streamlit page layouts and component configurations
4. **Compatibility issues**: AI fixed some complex compatibility issues with tensorflow.

#### What Humans Did:
- Reviewed all generated code for correctness
- Modified implementations to fit specific requirements
- Integrated AI-generated modules into unified architecture
- Tested functionality thoroughly

### 2.2 Refactoring

#### What AI Did:
- Converted Jupyter notebook code to modular Python scripts
- Reorganized code into logical modules and packages
- Suggested improvements for code structure
- Applied PEP 8 style guidelines

#### Specific Examples:
1. **Notebook Conversion**: Transformed monolithic notebook cells into reusable functions
2. **Module Organization**: Created `utils/`, `xai_methods/`, and proper package structure
3. **Code Cleanup**: Removed duplicate code and consolidated common logic

#### What Humans Did:
- Made architectural decisions about module boundaries
- Determined which code to keep, modify, or discard
- Ensured consistency across refactored modules
- Validated that refactored code maintained original functionality

### 2.3 Documentation

#### What AI Did:
- Generated README structure and content
- Created comprehensive technical report
- Wrote docstrings for functions and classes
- Produced usage examples and tutorials
- Formatted markdown documentation

#### Specific Examples:
1. **README.md**: AI generated installation instructions, usage guide, and feature descriptions
2. **Code Documentation**: AI wrote docstrings following NumPy/Google style

#### What Humans Did:
- Reviewed documentation for accuracy
- Added project-specific details (team names, TD group)
- Customized examples to match actual implementation
- Ensured technical accuracy of explanations

### 2.4 Debugging & Problem Solving

#### What AI Did:
- Suggested solutions for error messages
- Proposed alternative implementations
- Explained library usage and API details

#### Specific Examples:
1. **Import Errors**: AI helped resolve module import issues
2. **TensorFlow Warnings**: AI suggested solutions for deprecated API usage
3. **Streamlit State**: AI explained session state management patterns

#### What Humans Did:
- Exposed bugs
- Diagnosed root causes of issues
- Tested proposed solutions
- Adapted solutions to specific context
- Made final decisions on implementation approach

---

## 3. Conclusion

This project demonstrates responsible and transparent use of generative AI in an academic setting. The AI served as a productivity tool that:

- **Accelerated** routine coding tasks
- **Enabled** more time for learning and refinement
- **Improved** code quality and documentation
- **Did NOT replace** human understanding or decision-making

All team members:
1. ✅ Understand the entire codebase
2. ✅ Can explain technical decisions
3. ✅ Contributed meaningfully to the project
4. ✅ Used AI ethically and transparently
5. ✅ Gained genuine learning outcomes

We believe this approach aligns with modern software development practices while maintaining academic integrity and ensuring deep learning.

---

## 11. Team Signatures

By signing below, we affirm that:
- This AI usage statement is accurate and complete
- We understand all code in this project
- We have used AI ethically and transparently
- This represents our own work with AI assistance as described

**Team Members**:

1. REDON Guillaume - 08-01-2026
2. RENOIR Théo - 07-01-2026

---

**Last Updated**: January 2026
**TD Group**: DIA5
