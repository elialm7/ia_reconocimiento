"use client"

import type { FormEvent } from "react"

import { useChat } from "ai/react"
import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Send, Loader2, Bot, User } from "lucide-react"

export function MentalHealthChat() {
  const [inputValue, setInputValue] = useState("")
  const { messages, input, handleInputChange, handleSubmit, isLoading } = useChat({
    api: "/api/chat",
    initialMessages: [
      {
        id: "welcome-message",
        role: "assistant",
        content:
          "Hola, soy tu asistente de salud mental. ¿En qué puedo ayudarte hoy? Puedes hablarme sobre cómo te sientes o hacerme preguntas sobre salud mental.",
      },
    ],
    onFinish: () => {
      setInputValue("")
    },
  })

  const handleFormSubmit = (e: FormEvent<HTMLFormElement>) => {
    e.preventDefault()
    if (inputValue.trim()) {
      handleSubmit(e)
    }
  }

  return (
    <div className="flex flex-col h-[600px]">
      <ScrollArea className="flex-1 p-4">
        <div className="space-y-4">
          {messages.map((message) => (
            <div key={message.id} className={`flex ${message.role === "user" ? "justify-end" : "justify-start"}`}>
              <div
                className={`flex items-start gap-2 max-w-[80%] ${
                  message.role === "user"
                    ? "bg-teal-600 text-white rounded-l-lg rounded-tr-lg"
                    : "bg-gray-100 text-gray-800 rounded-r-lg rounded-tl-lg"
                } p-3`}
              >
                <div className="mt-0.5">
                  {message.role === "user" ? (
                    <User className="h-5 w-5 text-teal-100" />
                  ) : (
                    <Bot className="h-5 w-5 text-teal-600" />
                  )}
                </div>
                <div className="whitespace-pre-wrap">{message.content}</div>
              </div>
            </div>
          ))}
          {isLoading && (
            <div className="flex justify-start">
              <div className="bg-gray-100 text-gray-800 rounded-r-lg rounded-tl-lg p-3 flex items-center gap-2 max-w-[80%]">
                <Bot className="h-5 w-5 text-teal-600" />
                <Loader2 className="h-4 w-4 animate-spin text-teal-600" />
                <span>Pensando...</span>
              </div>
            </div>
          )}
        </div>
      </ScrollArea>

      <div className="border-t p-4">
        <form onSubmit={handleFormSubmit} className="flex gap-2">
          <Input
            placeholder="Escribe tu mensaje aquí..."
            value={inputValue}
            onChange={(e) => {
              setInputValue(e.target.value)
              handleInputChange(e)
            }}
            className="flex-1"
          />
          <Button type="submit" disabled={isLoading || !inputValue.trim()} className="bg-teal-600 hover:bg-teal-700">
            {isLoading ? <Loader2 className="h-4 w-4 animate-spin" /> : <Send className="h-4 w-4" />}
            <span className="sr-only">Enviar mensaje</span>
          </Button>
        </form>
      </div>
    </div>
  )
}
