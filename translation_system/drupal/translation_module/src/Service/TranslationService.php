<?php

namespace Drupal\translation_module\Service;

use GuzzleHttp\Client;
use GuzzleHttp\Exception\GuzzleException;
use Drupal\Core\Config\ConfigFactoryInterface;

/**
 * Service for interacting with translation API.
 */
class TranslationService {

  /**
   * The HTTP client.
   *
   * @var \GuzzleHttp\Client
   */
  protected $httpClient;

  /**
   * The config factory.
   *
   * @var \Drupal\Core\Config\ConfigFactoryInterface
   */
  protected $configFactory;

  /**
   * Constructor.
   */
  public function __construct(Client $http_client, ConfigFactoryInterface $config_factory) {
    $this->httpClient = $http_client;
    $this->configFactory = $config_factory;
  }

  /**
   * Get the translation API endpoint URL.
   */
  protected function getApiEndpoint() {
    $config = $this->configFactory->get('translation_module.settings');
    return $config->get('api_endpoint') ?: 'http://localhost:8000';
  }

  /**
   * Translate text.
   */
  public function translate($text, $use_memory = TRUE, $use_terminology = TRUE) {
    $endpoint = $this->getApiEndpoint() . '/translate';
    
    try {
      $response = $this->httpClient->post($endpoint, [
        'json' => [
          'text' => $text,
          'use_memory' => $use_memory,
          'use_terminology' => $use_terminology,
        ],
      ]);

      $data = json_decode($response->getBody(), TRUE);
      return $data;
    } catch (GuzzleException $e) {
      throw new \Exception('Translation API error: ' . $e->getMessage());
    }
  }

  /**
   * Get translation memory statistics.
   */
  public function getMemoryStats() {
    $endpoint = $this->getApiEndpoint() . '/memory/stats';
    
    try {
      $response = $this->httpClient->get($endpoint);
      $data = json_decode($response->getBody(), TRUE);
      return $data;
    } catch (GuzzleException $e) {
      throw new \Exception('Memory stats API error: ' . $e->getMessage());
    }
  }

  /**
   * Get terminology statistics.
   */
  public function getTerminologyStats() {
    $endpoint = $this->getApiEndpoint() . '/terminology/stats';
    
    try {
      $response = $this->httpClient->get($endpoint);
      $data = json_decode($response->getBody(), TRUE);
      return $data;
    } catch (GuzzleException $e) {
      throw new \Exception('Terminology stats API error: ' . $e->getMessage());
    }
  }

}
