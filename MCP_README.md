

### What's Included

#### MCP Tools Available

1. **`query_documentation`** - Full RAG query using your agent
   - Search and get intelligent responses from your network documentation
   - Supports vendor-specific searches (aruba, cisco, juniper, or all)

2. **`search_vector_database`** - Direct vector similarity search
   - Raw access to your ChromaDB vector search
   - Choose specific collections and number of results

3. **`scrape_url`** - Scrape new documentation URLs
   - Add new content from network vendor documentation sites
   - Supports all your configured vendors

4. **`web_search`** - Web search for additional context
   - Search the web for supplementary network information
   - Configurable number of results

5. **`get_collection_stats`** - Database statistics
   - Get document counts and health of your vector collections
   - Monitor your documentation database

6. **`ingest_url_list`** - Batch URL processing
   - Process multiple URLs from your vendor link files
   - Controllable batch size for safe processing

#### MCP Resources Available

1. **Document Collections** - Access to your ChromaDB collections
   - `all_vendor_docs` - Combined documentation
   - `aruba_docs` - Aruba-specific documentation  
   - `cisco_docs` - Cisco-specific documentation
   - `juniper_docs` - Juniper-specific documentation

2. **URL Collections** - Your scraped URL lists
   - Aruba, Cisco, Juniper URL collections for reference

## Quick Start

### 1. Install Dependencies

```bash
pip install mcp
```

### 2. Test the MCP Server

Run the test script to verify everything works:

```bash
python test_mcp_server.py
```

### 3. Connect to Claude Desktop

#### Option A: Copy Configuration
Copy the contents of `mcp_config.json` to your Claude Desktop configuration file:

**Windows:** `%APPDATA%\Claude\claude_desktop_config.json`
**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`

#### Option B: Manual Configuration
Add this to your Claude Desktop config:

```json
{
  "mcpServers": {
    "r1-rag": {
      "command": "python",
      "args": ["mcp_server.py"],
      "cwd": "/f:/R1_rag",
      "env": {
        "PYTHONPATH": "/f:/R1_rag"
      }
    }
  }
}
```

**Important:** Update the `cwd` path to match your actual project directory.

### 4. Restart Claude Desktop

After updating the configuration, restart Claude Desktop. You should see a hammer icon (🔨) indicating tools are available.

## Usage Examples

Once connected to Claude Desktop, you can ask questions like:

- **"What are the current statistics of my network documentation database?"**
  - Uses `get_collection_stats` tool

- **"Search for VLAN configuration information in the documentation"**
  - Uses `query_documentation` tool with your full RAG pipeline

- **"Find documents about Aruba switch management"**
  - Uses `search_vector_database` tool with Aruba collection

- **"Scrape this new Cisco documentation URL: https://..."**
  - Uses `scrape_url` tool to add new content

- **"Process 5 URLs from the Juniper link list"**
  - Uses `ingest_url_list` tool for batch processing

## File Structure

```
R1_rag/
├── mcp_server.py           # Main MCP server implementation
├── test_mcp_server.py      # Test script for validation
├── mcp_config.json         # Claude Desktop configuration
├── MCP_README.md           # This documentation
└── requirements.txt        # Updated with MCP dependency
```

## Benefits of MCP Integration

1. **Universal Access** - Any MCP-compatible AI assistant can now use your R1-RAG system
2. **Secure** - All data stays on your local machine
3. **Modular** - Tools can be used individually or combined
4. **Extensible** - Easy to add new tools and capabilities
5. **Standardized** - Uses the open MCP protocol

## Troubleshooting

### Server Won't Start
- Check that all R1-RAG dependencies are installed
- Verify your environment variables are set correctly
- Run `python test_mcp_server.py` for detailed error messages

### Claude Can't Connect
- Verify the `cwd` path in your configuration matches your project directory
- Ensure Python is in your system PATH
- Check Claude Desktop logs for error messages

### Tools Not Working
- Ensure your ChromaDB database has been initialized
- Check that your environment variables (API keys, etc.) are set
- Verify your `.env` file is properly configured

## Next Phases

This completes **Phase 1** of the MCP integration. Future phases could include:

- **Phase 2:** MCP Client integration to connect to external MCP servers
- **Phase 3:** Enhanced agent with multi-MCP orchestration  
- **Phase 4:** Web interface showing MCP connections and usage

## Security Notes

- The MCP server only exposes read operations and controlled write operations (scraping/ingestion)
- All data processing happens locally on your machine
- No sensitive data is transmitted to external services through MCP
- You control which tools Claude can access through the MCP configuration

## Support

If you encounter issues:
1. Run the test script first: `python test_mcp_server.py`
2. Check the MCP server logs for error messages
3. Verify all dependencies are installed correctly
4. Ensure your R1-RAG system works independently before testing MCP integration 