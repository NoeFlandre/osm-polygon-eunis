Feature: EUNIS release workflow

  Scenario: Preview every shard without uploading
    Given a single-shard fake Hub release
    When I preview the release through the CLI
    Then the preview lists the shard and the Hub receives no uploads

  Scenario: Publish and verify the release
    Given a single-shard fake Hub release
    When I publish the release through the CLI
    Then the Hub receives the shard manifest and card
    And the CLI reports the verified revision

  Scenario: Repeating an unchanged release is a no-op
    Given a fake Hub release already published with the current reference
    When I publish the release through the CLI
    Then the CLI reports a no-op and the Hub receives no uploads

  Scenario: A changed reference version forces a publish
    Given a fake Hub release published with an older reference
    When I publish the release through the CLI
    Then the Hub receives the shard manifest and card
    And the CLI reports a newly verified revision

  Scenario: A polygon without overlap keeps null labels
    Given a polygon shard with no matching EUNIS geometry
    When I enrich the shard through the public API
    Then its EUNIS labels are null

  Scenario: A missing Hub token fails before connecting
    Given a release command without a Hub token
    When I publish the release through the CLI
    Then the command exits with a usage error without contacting the Hub
