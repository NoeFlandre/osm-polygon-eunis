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

  Scenario: A polygon outside the EUNIS reference extent stays unlabeled
    Given a polygon shard outside the EUNIS reference extent
    When I enrich the shard through the public API
    Then its EUNIS labels are null

  Scenario: Equal overlap ties use a stable EUNIS code
    Given a polygon shard with equal overlap from two EUNIS references
    When I enrich the shard through the public API
    Then the lower EUNIS code wins the tie

  Scenario: Invalid geometry is rejected during shard enrichment
    Given a source shard with a point geometry
    When I enrich the shard through the public API
    Then its EUNIS labels are null
