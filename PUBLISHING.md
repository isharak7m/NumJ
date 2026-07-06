# Publishing JNumj to Maven Central

## Prerequisites

- Java 21+
- GPG key (for signing artifacts)
- Sonatype Central Portal account (already configured)

## Configuration Files

### `gradle.properties` (not committed to git)

Create `gradle.properties` in the project root:

```properties
sonatypeUser=<your-token-username>
sonatypePassword=<your-token-password>
signing.keyId=<last-8-chars-of-gpg-key-id>
signing.password=<your-gpg-passphrase>
signing.secretKeyRingFile=<path-to-secring.gpg>
VERSION=0.1.0
```

See `gradle.properties.example` for the template.

### GPG Key Setup (one-time)

```powershell
# Generate key
gpg --full-gen-key
# Choose: RSA, 2048 bits, no expiry
# Use the same email as your Sonatype account

# List keys to get key ID
gpg --list-secret-keys --keyid-format LONG

# Export public key and upload to keyserver
gpg --armor --export <KEY_ID> > public-key.asc
gpg --keyserver keyserver.ubuntu.com --send-keys <KEY_ID>

# Export private key for signing
gpg --batch --pinentry-mode loopback --passphrase "<passphrase>" --export-secret-keys --output secring.gpg <KEY_ID>
```

## Publish a Release

```powershell
.\gradlew.bat publishAggregationToCentralPortal
```

This uses the NMCP (New Maven Central Publishing) plugin to:
1. Build and sign all artifacts
2. Bundle them into a ZIP
3. Upload to the Sonatype Central Portal API
4. Wait for validation and automatic publishing

## Verify

Check the published artifact at: https://central.sonatype.com/search?q=jnumj

## Consumer Dependency

```kotlin
// Gradle
implementation("io.github.isharak7m:jnumj:0.1.0")
```

```xml
<!-- Maven -->
<dependency>
    <groupId>io.github.isharak7m</groupId>
    <artifactId>jnumj</artifactId>
    <version>0.1.0</version>
</dependency>
```
