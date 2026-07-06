# Publishing JNumj to Maven Central

## Prerequisites

- Java 21+
- GPG key (for signing artifacts)

## Step-by-Step

### 1. Register Namespace (one-time)

1. Go to https://central.sonatype.com
2. Sign in with your email (already done)
3. Click **"Publish"** in the left sidebar
4. Click **"Register Namespace"**
5. Enter: `io.github.jnumj`
6. Click **"Add"**

Sonatype will verify you own the domain. Since it's a GitHub namespace (`io.github.<username>`), this is automatic — just confirm.

### 2. Generate a User Token (one-time)

1. In the top-right, click your profile → **"View User Token"**
2. Click **"Generate User Token"**
3. Save the **username** and **password** shown — you'll never see them again

### 3. Generate a GPG Key (one-time)

```powershell
# Install GPG if not installed (https://gnupg.org/download/)
gpg --full-gen-key
# Choose: RSA, 4096 bits, no expiry
# Enter your name and email (must match your Sonatype account)
# Set a passphrase (remember it!)
```

List your key:
```powershell
gpg --list-secret-keys --keyid-format=long
```

You'll see output like:
```
sec   rsa4096/ABC123DEF456 2026-01-01
```

The part after `rsa4096/` is your **key ID** (e.g. `ABC123DEF456`).

Upload it to a keyserver:
```powershell
gpg --keyserver keyserver.ubuntu.com --send-keys ABC123DEF456
```

Export the private key (you'll need it for Gradle):
```powershell
gpg --armor --export-secret-keys ABC123DEF456 > private-key.asc
```

### 4. Create `gradle.properties`

Create `C:\projects\numj\gradle.properties`:

```properties
# Credentials from step 2
sonatypeUser=<your-token-username>
sonatypePassword=<your-token-password>

# GPG from step 3 (use your key ID and passphrase)
signing.keyId=ABC123DEF456
signing.password=<your-gpg-passphrase>
signing.secretKeyRingFile=C:/path/to/private-key.asc

# Optional: publish version
version=0.1.0
```

**Security note**: Never commit `gradle.properties` to git. The `.gitignore` already excludes it, but double-check.

### 5. Update `build.gradle.kts`

I'll update the build file with the correct publishing configuration — the current one uses the old Maven Central URL.

### 6. Publish

```powershell
.\gradlew.bat publish
```

### 7. Release on Sonatype

1. Go to https://central.sonatype.com → **"Publish"** → **"Staging Repositories"**
2. Find your repository, click **"Close"** (this validates it)
3. If it passes, click **"Publish"**

Within ~15 minutes, it's live on Maven Central. Anyone can use:

```kotlin
dependencies { implementation("io.github.jnumj:jnumj:0.1.0") }
```

### 8. Future Updates

For subsequent releases:
1. Update `version` in `gradle.properties`
2. Run `./gradlew publish`
3. Release on Sonatype (same as step 7)

---

Want me to update `build.gradle.kts` with the correct Sonatype portal configuration and signing plugin now?
