import discord
from discord.ext import commands

intents = discord.Intents.all()
client = commands.Bot(command_prefix = '!',intents=intents)

@client.event
async def on_ready():
    print("The bot is now ready for use!")
    print(".............................")


@client.command()
async def hello(ctx):
    await ctx.send("Hello, I am Elia. Is there anything I can help?")

client.run('')