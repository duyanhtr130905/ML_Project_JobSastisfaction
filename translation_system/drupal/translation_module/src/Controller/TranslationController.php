<?php

namespace Drupal\translation_module\Controller;

use Drupal\Core\Controller\ControllerBase;
use Symfony\Component\DependencyInjection\ContainerInterface;
use Drupal\translation_module\Service\TranslationService;
use Symfony\Component\HttpFoundation\JsonResponse;
use Symfony\Component\HttpFoundation\Request;

/**
 * Controller for translation operations.
 */
class TranslationController extends ControllerBase {

  /**
   * The translation service.
   *
   * @var \Drupal\translation_module\Service\TranslationService
   */
  protected $translationService;

  /**
   * Constructor.
   */
  public function __construct(TranslationService $translation_service) {
    $this->translationService = $translation_service;
  }

  /**
   * {@inheritdoc}
   */
  public static function create(ContainerInterface $container) {
    return new static(
      $container->get('translation_module.translation_service')
    );
  }

  /**
   * Main translation interface page.
   */
  public function translationPage() {
    return [
      '#theme' => 'translation_interface',
      '#source_text' => '',
      '#target_text' => '',
      '#translation_info' => [],
      '#attached' => [
        'library' => [
          'translation_module/translation_interface',
        ],
      ],
    ];
  }

  /**
   * API endpoint to translate text.
   */
  public function translateText(Request $request) {
    $data = json_decode($request->getContent(), TRUE);
    
    if (empty($data['text'])) {
      return new JsonResponse([
        'error' => 'No text provided'
      ], 400);
    }

    $text = $data['text'];
    $use_memory = $data['use_memory'] ?? TRUE;
    $use_terminology = $data['use_terminology'] ?? TRUE;

    try {
      $result = $this->translationService->translate($text, $use_memory, $use_terminology);
      return new JsonResponse($result);
    } catch (\Exception $e) {
      return new JsonResponse([
        'error' => $e->getMessage()
      ], 500);
    }
  }

  /**
   * Get translation memory statistics.
   */
  public function memoryStats() {
    try {
      $stats = $this->translationService->getMemoryStats();
      return new JsonResponse($stats);
    } catch (\Exception $e) {
      return new JsonResponse([
        'error' => $e->getMessage()
      ], 500);
    }
  }

}
