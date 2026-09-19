Feature: EUNIS enrichment

  Scenario: Enrich a geometry shard by actual polygon overlap
    Given a source shard with one polygon and two overlapping EUNIS geometries
    When I run the shard enrichment with batch size 1
    Then the source columns and row count are unchanged
    And the larger actual intersection supplies the label
    And the stored overlap percentage is the intersection percentage

  Scenario: Preserve shared Wikidata tables
    Given a Wikidata source file list with polygon and document tables
    When I build the enrichment manifest
    Then only polygon tables are changed
    And document tables remain shared

  Scenario: Publish a static EUNIS map and distribution table in the card
    Given a completed label summary
    When I build the dataset card
    Then the card contains a static map and percentage table
