import streamlit as st

# Page title and header
st.markdown(
    "<h1 style='text-align: center;'>AI Content Summarizer 🤖</h1>", 
    unsafe_allow_html=True
)
st.markdown(
    "<h3 style='text-align: center;'>Your Intelligent Content Analysis Assistant</h3>", 
    unsafe_allow_html=True
)

# Welcome message
st.markdown("""
<div style='text-align: center;'>
    <h2>Welcome! 👋</h2>
    <p>I'm your AI-powered content summarization assistant. I can help you:</p>
    <ul style='display: inline-block; text-align: left;'>
        <li>📰 Summarize news articles and answer questions about them</li>
        <li>🎬 Summarize YouTube videos and explain their content</li>
        <li>📄 Summarize documents (PDF, TXT, MD) and extract key insights</li>
    </ul>
    <p><strong>Get started by selecting a feature from the navigation bar above!</strong></p>
</div>
""", unsafe_allow_html=True)

# Feature cards
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    <div style='background-color:#1a1a1a; padding:20px; border-radius:10px; color:#f0f0f0;'>
        <h3 style='color:#4dabf7;'>📰 News Articles</h3>
        <p>Paste a news article URL to get:</p>
        <ul>
            <li>Concise summary</li>
            <li>Key points extraction</li>
            <li>Q&A about the content</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div style='background-color:#1a1a1a; padding:20px; border-radius:10px; color:#f0f0f0;'>
        <h3 style='color:#ff6b6b;'>🎬 YouTube Videos</h3>
        <p>Enter a YouTube URL to get:</p>
        <ul>
            <li>Video transcript summary</li>
            <li>Key takeaways</li>
            <li>Q&A about the video content</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div style='background-color:#1a1a1a; padding:20px; border-radius:10px; color:#f0f0f0;'>
        <h3 style='color:#51cf66;'>📄 Documents</h3>
        <p>Upload PDF, TXT, or MD files to get:</p>
        <ul>
            <li>Document summary</li>
            <li>Key insights extraction</li>
            <li>Q&A about the document</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# Footer
st.divider()
st.caption("Built with ❤️ using Streamlit, LangChain, and OpenAI")
