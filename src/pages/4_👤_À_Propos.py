import streamlit as st

def main():
    st.markdown(
        "<h1 style='text-align: center; color: #4CAF50;'>À Propos de l'Auteur 📝</h1>",
        unsafe_allow_html=True
    )

    # Ajout d'un expander pour l'auteur
    with st.expander("Auteur", True):
        c_1, c_2 = st.columns([1, 2])
        with c_1:
            st.image("src/About the author.png", caption="S. Abraham Z. KOLOBOE", use_container_width=True)
        with c_2:
            st.header("**S. Abraham Z. KOLOBOE**")
            st.markdown("""
                *:blue[Data Scientist | Ingénieur en Mathématiques et Modélisation]*  
                Bonjour,  
                Je suis Abraham, un Data Scientist et Ingénieur en Mathématiques et Modélisation.  
                Mon expertise se situe dans les domaines des sciences de données et de l'intelligence artificielle.  
                Avec une approche technique et concise, je m'engage à fournir des solutions efficaces et précises dans mes projets.
                
                * **Email** : [abklb27@gmail.com](mailto:abklb27@gmail.com)
                * **WhatsApp** : +229 91 83 84 21
                * **LinkedIn** : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)
            """, unsafe_allow_html=True)

    # Sidebar content
    with st.sidebar:
        st.markdown("""
            ## Auteur
            *:blue[Abraham KOLOBOE]*  
            * **Email** : [abklb27@gmail.com](mailto:abklb27@gmail.com)  
            * **WhatsApp** : +229 91 83 84 21  
            * **LinkedIn** : [Abraham KOLOBOE](https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning)  
        """, unsafe_allow_html=True)
    # Footer avec emojis et style
    st.markdown("---")
    st.markdown(
        """
        <style>
        .footer {
            text-align: center;
            color: #4CAF50;
            font-size: 16px;
            margin-top: 20px;
        }
        .footer a {
            color: #4CAF50;
            text-decoration: none;
        }
        .footer a:hover {
            text-decoration: underline;
        }
        </style>
        <p class='footer'>💡 Dashboard créé avec ❤️ par <a href='https://www.linkedin.com/in/abraham-zacharie-koloboe-data-science-ia-generative-llms-machine-learning/' target='_blank'>Abraham KOLOBOE</a></p>
        """,
        unsafe_allow_html=True
    )
if __name__ == "__main__":
    main()
