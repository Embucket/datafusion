# Multi-stage Docker build for datafusion-cli
# Build stage
FROM rust:1.86-slim as builder

# Install system dependencies needed for building
RUN apt-get update && apt-get install -y \
    pkg-config \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /usr/src/datafusion

# Copy Cargo files first for better layer caching
COPY Cargo.toml Cargo.lock ./
COPY datafusion/ ./datafusion/
COPY datafusion-cli/ ./datafusion-cli/
COPY datafusion-examples/ ./datafusion-examples/
COPY test-utils/ ./test-utils/
COPY benchmarks/ ./benchmarks/

# Build the datafusion-cli binary in release mode
RUN cargo build --release --bin datafusion-cli

# Runtime stage
FROM debian:bookworm-slim

# Install runtime dependencies
RUN apt-get update && apt-get install -y \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Copy the binary from builder stage
COPY --from=builder /usr/src/datafusion/target/release/datafusion-cli /usr/local/bin/datafusion-cli

# Set the binary as executable
RUN chmod +x /usr/local/bin/datafusion-cli

# Set datafusion-cli as the entrypoint
ENTRYPOINT ["datafusion-cli"]