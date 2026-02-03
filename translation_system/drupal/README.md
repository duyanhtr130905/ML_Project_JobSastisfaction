# Drupal Translation Module Configuration

This directory contains configuration and integration files for Drupal CMS.

## Overview

The Drupal module integrates the EN-VI translation system with Drupal's content management capabilities, allowing users to:

1. Translate content directly from the Drupal admin interface
2. Manage translation memory through Drupal
3. Configure terminology and knowledge graph settings
4. View translation analytics and performance metrics

## Structure

```
drupal/
├── translation_module/
│   ├── translation_module.info.yml
│   ├── translation_module.module
│   ├── src/
│   │   ├── Controller/
│   │   │   └── TranslationController.php
│   │   ├── Form/
│   │   │   └── TranslationConfigForm.php
│   │   └── Service/
│   │       └── TranslationService.php
│   └── templates/
│       └── translation-interface.html.twig
└── README.md
```

## Installation

1. Copy the `translation_module` directory to your Drupal `modules/custom/` directory
2. Enable the module via Drupal admin or drush:
   ```bash
   drush en translation_module -y
   ```
3. Configure the translation API endpoint in the module settings
4. Grant appropriate permissions to user roles

## Configuration

Navigate to `Configuration > Translation System` to configure:

- Translation API endpoint URL
- Enable/disable translation memory
- Enable/disable terminology standardization
- API authentication settings

## Usage

### From Drupal Admin

1. Navigate to `Content > Translate`
2. Enter English text to translate
3. Click "Translate" to get Vietnamese translation
4. Review and edit translation if needed
5. Save to content or translation memory

### Via REST API

The module exposes REST endpoints:

- `POST /api/translation/translate` - Translate text
- `GET /api/translation/memory` - Get translation memory
- `POST /api/translation/terminology` - Add terminology

## Requirements

- Drupal 9.x or 10.x
- PHP 7.4 or higher
- Translation System API running and accessible

## Support

For issues or questions, contact the development team.
